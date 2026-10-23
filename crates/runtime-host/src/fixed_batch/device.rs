use super::diagnostic::{self, Budget, Description, Options};
use super::record::{Capture, CapturedRow, Frame, Golden, SampleBits, create, metadata};
use super::trace::{MIXED_DECODE, Trace};
use crate::executor::{
    BatchWireCapabilityV1, BundleExecutionV1, DeviceLogitsSamplerV1, SamplingDispatch,
    batch_wire_capability_v1, write_batch_request_v1, write_state_slots_v1,
};
use crate::services::HostServicesV1Backend;
use runtime_bundle_abi::{
    BundleStatusV1, HostServiceResultV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
};
use runtime_engine::{ModelBatchItem, RequestId, SampledRow};
use serde::Deserialize;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Backend = HostServicesV1Backend<crate::cuda::CudaHostDeviceV1>;

fn memory(backend: &Backend) -> Result<Value> {
    use crate::probe::{HostDeviceMemoryUsageV1, HostDeviceRankMemoryUsageV1};
    backend.with_device(|device| {
        let report = device.rank_memory_v1()?;
        Ok(json!({
            "measurement": report.measurement().as_str(),
            "current_allocation_bytes": device.current_device_memory_bytes(),
            "peak_allocation_bytes": device.peak_device_memory_bytes(),
            "kv_pages": backend.serving_kv_pages(),
            "ranks": report.ranks().iter().map(|r| json!({
                "rank": r.rank(), "free_bytes": r.free_bytes(), "total_bytes": r.total_bytes(),
                "allocated_bytes": r.allocated_bytes(), "peak_allocated_bytes": r.peak_allocated_bytes(),
            })).collect::<Vec<_>>(),
        }))
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    bundle: PathBuf,
    /// The checkpoint the package's weight plans load.
    checkpoint: PathBuf,
    topology: String,
    trace: PathBuf,
    output: PathBuf,
    reference: Option<PathBuf>,
    control: Option<PathBuf>,
    concurrency: usize,
    verify: bool,
    #[serde(default)]
    prefill_only: bool,
    #[serde(default)]
    draft_tokens: Option<usize>,
    owned: bool,
    source: Value,
    #[serde(default)]
    verify_rows_per_item: Option<usize>,
    #[serde(default)]
    max_sequences: Option<usize>,
    #[serde(default)]
    trace_input: bool,
    #[serde(default)]
    diagnostic: Option<Options>,
    /// Records an explicit mixed decode trace at `max_sequences`, without a
    /// reference: arms are compared offline.
    #[serde(default)]
    mixed: bool,
    /// KV pages of a mixed decode replay, the same for every compared arm.
    #[serde(default)]
    kv_pages: Option<usize>,
}

struct RecordingSampler {
    backend: Rc<Backend>,
    capture: RefCell<Capture>,
    budget: Option<RefCell<Budget>>,
}
impl RecordingSampler {
    fn raw(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<()> {
        let raw = view.raw();
        if items.len() != raw.row_count as usize {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if let Some(budget) = &self.budget {
            budget
                .borrow_mut()
                .reserve(raw.element_type, raw.live_width, items.len())
                .map_err(refusal)?;
        }
        for (offset, item) in items.iter().enumerate() {
            let bytes = self.backend.read_logits_view_row(view, offset as u32)?;
            let row = CapturedRow {
                rank: raw.rank,
                element_type: raw.element_type,
                width: raw.live_width,
                bytes,
                metadata: metadata(item, dispatch),
                sample: None,
            };
            self.capture
                .borrow_mut()
                .insert(raw.first_item_index as usize + offset, row)
                .map_err(refusal)?;
        }
        Ok(())
    }
    fn sampled(&self, view: OwnedLogitsViewV1, rows: &[SampledRow]) -> HostServiceResultV1<()> {
        self.capture
            .borrow_mut()
            .samples(view.raw().first_item_index as usize, rows)
            .map_err(refusal)
    }
}
fn refusal(error: Box<dyn std::error::Error>) -> BundleStatusV1 {
    eprintln!("fixed trace recorder: {error}");
    BundleStatusV1::HostError
}
impl DeviceLogitsSamplerV1 for RecordingSampler {
    fn sample_view(
        &self,
        _: OwnedLogitsViewV1,
        _: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        Err(BundleStatusV1::InvalidState)
    }
    fn sample_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        self.raw(view, items, dispatch)?;
        let rows = self
            .backend
            .sample_view_with_dispatch(view, items, dispatch)?;
        self.sampled(view, &rows)?;
        Ok(rows)
    }
    fn issue_view_with_dispatch(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: SamplingDispatch,
    ) -> HostServiceResultV1<bool> {
        self.raw(view, items, dispatch)?;
        if !self
            .backend
            .issue_view_with_dispatch(view, items, dispatch)?
        {
            return Err(BundleStatusV1::Unsupported);
        }
        Ok(true)
    }
    fn collect_view(
        &self,
        view: OwnedLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        let rows = self.backend.collect_view(view, items)?;
        self.sampled(view, &rows)?;
        Ok(rows)
    }
}

#[test]
#[ignore = "requires a direct package, its checkpoint, a recorded trace configuration, and an exclusive TP4 slot"]
fn replay_installed_trace_gpu() -> Result<()> {
    let config: Config =
        serde_json::from_reader(File::open(std::env::var("CHOPINFER_BATCH_REPLAY_CONFIG")?)?)?;
    // One distinct ordinal per rank.
    let devices = std::env::var("CHOPINFER_KERNEL_GPUS")?
        .split(',')
        .map(str::parse::<i32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let rank_count = devices.len();
    let distinct = devices.iter().collect::<BTreeSet<_>>().len();
    if ![2, 4, 8].contains(&rank_count) || distinct != rank_count {
        return Err("two, four or eight ranks on distinct devices required".into());
    }
    let compare_packages = config.verify_rows_per_item.is_some();
    let max_sequences = config.max_sequences.unwrap_or(config.concurrency);
    let mixed = config.mixed;
    if mixed
        && (!config.trace_input
            || config.verify
            || config.owned
            || config.prefill_only
            || config.draft_tokens.is_some()
            || compare_packages
            || config.reference.is_some()
            || config.control.is_some()
            || config.diagnostic.is_some()
            || config.kv_pages.is_none()
            || config.max_sequences.is_none_or(|m| m < config.concurrency))
    {
        return Err(
            "a mixed decode replay records an explicit trace at explicit capacity and pages".into(),
        );
    }
    if !mixed
        && ((config.prefill_only
            && (config.verify
                || config.owned
                || compare_packages
                || config.draft_tokens.is_some()))
            || if config.prefill_only {
                !(1..=32).contains(&config.concurrency)
            } else {
                ![1, 16, 32].contains(&config.concurrency)
            }
            || (config.prefill_only && (max_sequences < config.concurrency || max_sequences > 32))
            || config.kv_pages.is_some()
            || (config.verify
                && config.concurrency != 16
                && !compare_packages
                && config.draft_tokens.is_none())
            || (compare_packages && (!config.verify || config.owned))
            || (!compare_packages && !config.prefill_only && config.max_sequences.is_some())
            || config.draft_tokens.is_some_and(|depth| {
                !config.verify || compare_packages || !(1..=3).contains(&depth)
            }))
    {
        return Err("unsupported trace concurrency".into());
    }
    let arguments = vec![
        "serve".into(),
        "--bundle".into(),
        config.bundle.to_string_lossy().into_owned(),
    ];
    let crate::HostCommandV1::Serve {
        selection: crate::preparation::ServeSelection::Bundle(selection),
        host,
        ..
    } = crate::parse_host_command_v1(&arguments)?
    else {
        unreachable!()
    };
    crate::direct::select_checkpoint_v1(&config.checkpoint)?;
    let loaded = crate::load_selected_topology_bundle_v1(&selection, &host, &config.topology)?;
    let deployment =
        crate::deployment_from_bundle_v1(loaded, runtime_server::ServerConfig::default())?;
    let vocabulary = deployment.policy().vocabulary().get();
    let boundaries = crate::generation::boundary_token_ids_v1(&deployment.policy());
    let tokenizer = Arc::new(deployment.into_tokenizer());
    let trace = Trace::select(
        &config.trace,
        config.reference.is_some() || config.trace_input,
        || {
            let prompts = (0..config.concurrency)
                .map(|slot| -> Result<_> {
                    let text = format!(
                    "Explain how the number {} appears in a short story about trees and rivers. ",
                    slot + 10
                )
                .repeat(32);
                    let tokens = tokenizer.encode(&text, &[])?;
                    if tokens.len() < 128 {
                        return Err("prompt recipe is too short".into());
                    }
                    Ok(tokens[..128].to_vec())
                })
                .collect::<Result<Vec<_>>>()?;
            let trace = if let Some(rows) = config.verify_rows_per_item {
                Trace::construct_verify(prompts, vocabulary, rows)?
            } else if let Some(depth) = config.draft_tokens {
                Trace::construct_fixed_depth(prompts, vocabulary, depth)
            } else {
                Trace::construct(prompts, vocabulary, config.verify)
            };
            Ok(trace)
        },
    )?;
    trace.validate()?;
    if let Some(depth) = config.draft_tokens
        && trace
            .steps
            .iter()
            .skip(1)
            .any(|step| step.items.iter().any(|item| item.drafts != depth))
    {
        return Err("recorded trace has a different fixed draft depth".into());
    }
    if trace.concurrency != config.concurrency
        || (trace.version == MIXED_DECODE) != mixed
        || trace.vocabulary != vocabulary
        || trace
            .steps
            .iter()
            .any(|s| s.items.iter().any(|i| i.drafts > 0))
            != config.verify
    {
        return Err("configured case differs from recorded trace".into());
    }
    if config
        .verify_rows_per_item
        .is_some_and(|rows| !trace.matches_verify_rows(rows))
        || (!compare_packages && !mixed && trace.version != if config.prefill_only { 4 } else { 1 })
    {
        return Err("configured verify geometry differs from recorded trace".into());
    }
    let maximum_verify_tokens = if config.prefill_only {
        0
    } else if compare_packages {
        trace.verify_capacity(max_sequences)?
    } else if config.draft_tokens.is_some() {
        trace.steps.iter().map(|step| step.rows()).max().unwrap()
    } else {
        32
    };
    let manifest = serde_json::from_slice::<Value>(&loaded.manifest().to_canonical_json()?)?;
    let mut identity = json!({"format":"fixed-batch-logits-v1","source":config.source,
        "manifest":manifest,
        "topology":config.topology,"trace":trace});
    if compare_packages || config.prefill_only {
        // Package identity is evidence, while the host and fixed trace are invariants.
        identity = json!({"format":"fixed-batch-packages-v4","source":config.source,
            "topology":config.topology,"ranks":rank_count,"fold":false,"trace":trace,
            "max_sequences":max_sequences,"maximum_verify_tokens":maximum_verify_tokens});
    }
    let page_tokens = loaded.manifest().kv_layout().page_tokens() as usize;
    let quota = if mixed {
        trace.final_tokens()
    } else if config.prefill_only {
        trace.steps[0]
            .items
            .iter()
            .map(|i| i.inputs.len())
            .max()
            .unwrap()
            .max(160)
    } else {
        160
    };
    let pages = config
        .kv_pages
        .unwrap_or(max_sequences * quota.div_ceil(page_tokens) + max_sequences);
    if mixed {
        // Arms at other capacities replay the same trace bytes and pages.
        identity = json!({"format":"fixed-batch-mixed-v1","source":config.source,
            "topology":config.topology,"ranks":rank_count,
            "trace_sha256":diagnostic::digest(&config.trace)?,
            "max_sequences":max_sequences,"kv_pages":pages});
        let limits = loaded
            .manifest()
            .batch_limits()
            .ok_or("the package declares no batch limits")?;
        let served = crate::served_sequences_v1(
            loaded
                .manifest()
                .resource_requirements()
                .required_capabilities(),
            Some(limits),
            Some(max_sequences as u32),
        )?;
        if served != Some(max_sequences as u32)
            || trace.prefill_tokens() > limits.maximum_prefill_tokens() as usize
            || pages < config.concurrency * quota.div_ceil(page_tokens)
        {
            return Err("the package or KV pages cannot hold the mixed decode trace".into());
        }
    }
    if std::env::var_os("CHOPINFER_BATCH_REPLAY_DRY_RUN").is_some_and(|v| v == "1") {
        // Everything above runs without a device; nothing is written.
        println!(
            "fixed trace dry run: {}",
            json!({"identity":identity,"steps":trace.steps.len(),
                "rows":trace.steps.iter().map(|s| s.rows()).sum::<usize>(),
                "prefill_tokens":trace.prefill_tokens(),"quota":quota,"kv_pages":pages,
                "page_tokens":page_tokens,"vocabulary":vocabulary})
        );
        return Ok(());
    }
    if config.owned {
        let control: Value = serde_json::from_reader(File::open(
            config
                .control
                .as_ref()
                .ok_or("owned arm requires gathered repeat receipt")?
                .join("receipt.json"),
        )?)?;
        if control["passed"] != true
            || control["identity"] != identity
            || control["reference"] != json!(config.reference)
        {
            return Err("gathered/gathered identity gate did not pass for this trace".into());
        }
    }
    let mut golden = if let Some(options) = &config.diagnostic {
        options.validate_environment(std::env::vars())?;
        let estimate = options
            .limits
            .estimate(trace.steps.iter().map(|s| s.rows()), vocabulary as u32)?;
        eprintln!("fixed trace maximum capture: {estimate} bytes");
        let metadata = trace.metadata_bound() + serde_json::to_vec(&identity)?.len() as u64 + 1;
        options.limits.check_metadata(metadata)?;
        eprintln!("fixed trace maximum frame metadata: {metadata} bytes");
        let description = Description {
            invariants: json!({"host":config.source,"trace":diagnostic::digest(&config.trace)?,
                "vocabulary":vocabulary,"topology":config.topology,"ranks":rank_count,"fold":false,
                "capacity":{"sequences":max_sequences,"verify_tokens":maximum_verify_tokens},
                "sampling":"fixed-batch-v1"}),
            manifest: diagnostic::manifest_identity(&manifest),
            environment: options.environment.clone(),
            source_checkpoint: None,
            declared_differences: options.declared_differences.clone(),
            reproduction: options.reproduction.clone(),
            expected_steps: trace.steps.len(),
            expected_rows: trace.steps.iter().map(|s| s.rows()).sum(),
            limits: options.limits.clone(),
            context: json!({"trace":config.trace,"manifest_sha256":runtime_weights::sha256_bytes(&loaded.manifest().to_canonical_json()?),
                "configured_graph_switches":options.environment,"observed_graph_execution":null,
                "bucket_observation":"calculated padding only","source_checkpoint":"unavailable in this package format"}),
        };
        Golden::open_diagnostic(
            &config.output,
            config.reference.as_deref(),
            config.control.as_deref(),
            &identity,
            description,
        )?
    } else {
        Golden::open(&config.output, config.reference.as_deref(), &identity)?
    };
    let grammar = crate::grammar::GrammarServiceV1::new(tokenizer, vocabulary, boundaries)
        .compile(
            &Arc::new(runtime_grammar::regex_grammar("[a-z ]+")?),
            &|| false,
        )?;
    let constraint = runtime_engine::RowConstraint::new(runtime_grammar::ConstraintState::new(
        grammar,
        None,
        Vec::new(),
    ));
    if constraint
        .mask()
        .is_none_or(|mask| mask.iter().all(|&word| word == 0))
    {
        return Err("grammar has no allowed token mask".into());
    }
    let maximum_drafts = trace
        .steps
        .iter()
        .flat_map(|s| &s.items)
        .map(|i| i.drafts)
        .max()
        .unwrap_or(0)
        .max(2);
    let mut slots = if loaded.manifest().recurrent_state_layouts().is_empty() {
        None
    } else {
        Some(crate::recurrent::StateSlotPoolV1::new(
            loaded.manifest(),
            max_sequences,
            maximum_drafts + 1,
        )?)
    };
    for slot in 0..config.concurrency {
        if let Some(pool) = &mut slots {
            assert_eq!(pool.allocate(RequestId::new(slot as u64 + 1))?, slot as u64);
        }
    }
    let mut kv = runtime_kv::KvPool::new(
        runtime_kv::KvGeometry::new(page_tokens, pages)?,
        max_sequences,
    )?;
    let handles = (0..config.concurrency)
        .map(|_| kv.create_sequence(quota))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut speculation = runtime_bundle_abi::BundleRawSpeculationV1::disabled();
    if config.verify {
        speculation.maximum_draft_tokens = maximum_drafts as u32;
        speculation.maximum_verify_tokens = maximum_verify_tokens as u32;
    }
    let (backend, mut instance) = loaded.create_instance_with_serving_options(
        crate::cuda::CudaHostDeviceV1::new()?,
        &config.topology,
        devices,
        Some(crate::ServingKvPolicyV1 {
            sampling_vocabulary: Some(vocabulary),
            maximum_context_tokens: None,
            state_reserved_bytes: slots.as_ref().map_or(0, |p| p.reserved_bytes_per_rank()),
            utilization_millionths: 1_000_000,
            maximum_pages: Some(pages as u64),
            ..Default::default()
        }),
        speculation,
        Some(max_sequences as u32),
        None,
    )?;
    let sampler = Rc::new(RecordingSampler {
        backend: backend.clone(),
        capture: RefCell::new(Capture::default()),
        budget: config
            .diagnostic
            .as_ref()
            .map(|o| RefCell::new(Budget::new(o.limits.clone()))),
    });
    let memory_after_create = memory(&backend)?;
    let mut rank_rows: BTreeMap<u32, usize> = BTreeMap::new();
    let mut boundary_splits = 0;
    let result = (|| -> Result<()> {
        for (index, step) in trace.steps.iter().enumerate() {
            let items = step.model_items(Some(&constraint));
            let appends = step
                .items
                .iter()
                .map(|item| {
                    (
                        handles[item.slot],
                        if item.prefill {
                            item.inputs.len()
                        } else {
                            item.drafts + 1
                        },
                    )
                })
                .collect::<Vec<_>>();
            let plan = kv.prepare_append_batch(&appends)?;
            backend
                .prepare_serving_batch(&plan)
                .map_err(|s| format!("KV preparation: {s:?}"))?;
            let wire = if config.verify || slots.is_some() {
                BatchWireCapabilityV1::V3MediaSpeculative
            } else {
                batch_wire_capability_v1(loaded.manifest())
            };
            let mut request = Vec::new();
            if let Some(slots) = &slots {
                write_state_slots_v1(&mut request, &items, slots, 0, |request, trailing| {
                    write_batch_request_v1(request, &items, &plan, wire, trailing)
                })?;
            } else {
                write_batch_request_v1(&mut request, &items, &plan, wire, 0)?;
            }
            let limits = ResultSinkLimitsV1::new(
                step.rows() as u64,
                vocabulary as u64,
                (step.rows() * vocabulary) as u64,
            );
            let emitted =
                instance.execute_with_sampling(&request, limits, &items, Some(sampler.clone()))?;
            if !emitted.rows.is_empty() || !emitted.prompt_view_rows.is_empty() {
                return Err("trace bypassed device sampling".into());
            }
            let rows = sampler.capture.borrow_mut().take(step.rows())?;
            let samples = rows
                .iter()
                .map(|row| row.sample.clone().unwrap())
                .collect::<Vec<_>>();
            if emitted
                .sampled
                .iter()
                .map(SampleBits::from)
                .collect::<Vec<_>>()
                != samples
            {
                return Err("executor changed sampled row ordering".into());
            }
            let mut request_ranks: BTreeMap<u64, BTreeSet<u32>> = BTreeMap::new();
            for row in &rows {
                *rank_rows.entry(row.rank).or_default() += 1;
                request_ranks
                    .entry(row.sample.as_ref().unwrap().request)
                    .or_default()
                    .insert(row.rank);
            }
            if index > 0 {
                // Mixed replays run each package's default logits owners.
                if !compare_packages && !mixed {
                    super::trace::validate_row_owners(
                        &rows.iter().map(|row| row.rank).collect::<Vec<_>>(),
                        rank_count,
                        config.owned,
                    )?;
                }
                boundary_splits += request_ranks
                    .values()
                    .filter(|owners| owners.len() > 1)
                    .count();
            }
            let kept = step
                .items
                .iter()
                .map(|item| (item.slot as u64 + 1, item.kept))
                .collect::<Vec<_>>();
            let drafts = emitted
                .drafts
                .iter()
                .map(|d| (d.request_id, d.tokens.clone()))
                .collect();
            let frame = Frame::from_rows(
                request,
                kept,
                if step.items.iter().all(|item| item.prefill) {
                    0
                } else {
                    step.rows().next_power_of_two()
                },
                drafts,
                &rows,
            );
            if let Err(error) = golden.step(&frame, &rows) {
                if config.diagnostic.is_some() {
                    return Err(error);
                }
                serde_json::to_writer(create(&config.output.join("mismatch.json"))?, &frame)?;
                let mut bytes = create(&config.output.join("mismatch.bin"))?;
                for row in &rows {
                    std::io::Write::write_all(&mut bytes, &row.bytes)?;
                }
                return Err(error);
            }
            if let Some(budget) = &sampler.budget {
                budget.borrow_mut().next_step();
            }
            let partial = step
                .items
                .iter()
                .filter(|i| i.drafts > 0)
                .map(|i| (handles[i.slot], i.kept))
                .collect::<Vec<_>>();
            let kv_commit = kv.preflight_commit_prepared_kept(plan.id(), &partial, &[])?;
            if partial.is_empty() {
                instance.commit()?;
            } else {
                let records = step
                    .items
                    .iter()
                    .filter(|i| i.drafts > 0)
                    .map(|i| runtime_bundle_abi::BundleKeptRowsV1 {
                        struct_size: std::mem::size_of::<runtime_bundle_abi::BundleKeptRowsV1>()
                            as u32,
                        reserved: 0,
                        request_id: i.slot as u64 + 1,
                        kept_rows: i.kept as u32,
                        reserved_tail: 0,
                    })
                    .collect::<Vec<_>>();
                instance.commit_accepted(&records)?;
            }
            kv.apply_prepared_commit(kv_commit)?;
            eprintln!(
                "fixed trace step {index}: {} rows, {:?}, exact records",
                step.rows(),
                step.mode
            );
        }
        // Fixed K1/K3 items align with owner boundaries. Mixed and fixed K2
        // traces must still demonstrate an item split across owners.
        if config.owned
            && config.verify
            && matches!(config.draft_tokens, None | Some(2))
            && boundary_splits == 0
        {
            return Err("verify trace missed owner-boundary splits".into());
        }
        golden.finish(trace.steps.len())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = instance.cancel();
    }
    let memory_after_steps = memory(&backend)?;
    let destroyed = instance.destroy();
    if let Err(error) = &result
        && config.diagnostic.is_some()
        && !config.output.join("diagnostic.json").exists()
    {
        golden.publish("insufficient", &error.to_string())?;
    }
    result.map_err(|error| format!("fixed trace execution or comparison: {error}"))?;
    destroyed.map_err(|error| {
        format!(
            "fixed trace bundle destruction after {} completed steps: {error:?}",
            golden.steps
        )
    })?;
    if backend.live_resource_count() != 0 {
        return Err("bundle left host resources live".into());
    }
    let receipt = json!({"passed":true,"identity":identity,"reference":config.reference,
        "mixed":mixed,"kv_pages":pages,
        "memory_after_create":memory_after_create,"memory_after_steps":memory_after_steps,
        "manifest":manifest,"verify_rows_per_item":config.verify_rows_per_item,
        "max_sequences":max_sequences,"maximum_verify_tokens":maximum_verify_tokens,
        "steps":golden.steps,"rows":golden.rows,"bytes_recorded_or_compared":golden.bytes_checked,
        "rank_rows":rank_rows,"owner_boundary_splits":boundary_splits,"live_resources":0});
    serde_json::to_writer_pretty(create(&config.output.join("receipt.json"))?, &receipt)?;
    golden.publish(
        if config.reference.is_none() {
            "baseline"
        } else {
            "match"
        },
        "trace completed",
    )?;
    eprintln!(
        "fixed trace passed: {} steps, {} rows",
        golden.steps, golden.rows
    );
    Ok(())
}
