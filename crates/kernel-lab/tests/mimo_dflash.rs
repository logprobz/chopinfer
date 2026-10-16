//! Execute the bundle's DFlash drafter plans against the real-weight f64 oracle.
//!
//! A fixed context of synthetic target taps runs through the prefill-phase
//! taps and context projection on four ranks, then one verify-phase draft
//! block runs from a bonus token. The context is longer than the 2048-entry
//! circular cache, so the cache wraps and skips the chunk's oldest rows, and
//! each query's 1023-position window starts inside it. The context cache,
//! the draft logits and the drafted tokens are compared with
//! `model_mimo26::oracle::dflash`.
//!
//! `dflash_drafter_step_cost` times the drafter's share of a step with graph
//! replay: the taps and context projection every phase runs, and the draft
//! block of a verify phase, for batches up to the 32-row cap.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[path = "common/bundle_launch.rs"]
mod bundle_launch;
use bundle_launch::{allocate, bf16_bytes, launch, read, write};

#[path = "common/mimo_planned.rs"]
mod mimo_planned;
use mimo_planned::PlannedImages;
use model_mimo26_contract::{
    prepared::component_id,
    resources::ScratchBuffers,
    weight_plan::{DRAFTER_IMAGE_ROLE, Drafter, TEXT_IMAGE_ROLE},
};

use bundle_mimo26::dflash::program::{self, Peers, RankBuffers};
use bundle_mimo26::{
    contract::{DECODE_ENTRY_ROWS, Mimo},
    ops::{GatherRegions, LogitsBuffers, Ops, OwnPlan, Plan},
    pipeline::PreparedLaunch,
    program_lower::{self, DeviceInputs, RankStage},
    projection::{Choices, Prepared, Rank as Projections, rank_store},
    scratch::{COMM_BUCKETS, CommState, RankScratch},
    selection::Phase,
    sm120,
    switches::Switches,
    topology::Tp,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::select::{NanKeys, Ties, topk};
use kernel_lab::{LabResult, Session};
use model_mimo26::{
    catalog::{DrafterKind, precision_components},
    oracle::{ModelEnds, dflash::DflashOracle},
};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use runtime_cuda::DeviceBuffer;

const CONTEXT: usize = 2100;
const CACHE: usize = program::CACHE_TOKENS as usize;
const BLOCK: usize = 8;
const BONUS: u32 = 3_837;
const HIDDEN: usize = 4096;
const TOKENS: usize = 8192;
const VOCAB_PER_RANK: usize = 38_144;
/// The phase-1 row cap: the most rows a timed batch uses.
const TIMED_ROWS: usize = 32;

struct Phases {
    context: DeviceInputs,
    verify: DeviceInputs,
}

struct Rank {
    dflash: RankBuffers,
    /// The packaged component's drafter and head selections over the views.
    store: bundle_runtime::projection::Store,
    scratch: RankScratch,
    /// Hidden flags and epoch of the 8192-row and 8-row buckets, then one
    /// per timed batch.
    comm: Vec<CommState>,
    activation: Span,
    normalized: Span,
    workspace: Span,
    logits: Span,
    inputs: Phases,
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The DFlash drafter image and the target's shared embedding and head
/// (a text image of no layers), planned from `checkpoint`.
fn planned_images(checkpoint: &str) -> (PlannedImages, PlannedImages) {
    let checkpoint = Path::new(checkpoint);
    let drafter = PlannedImages::plan(
        checkpoint,
        DRAFTER_IMAGE_ROLE,
        4,
        None,
        Some(Drafter::Dflash),
    )
    .expect("planned DFlash image");
    let target = PlannedImages::plan(checkpoint, TEXT_IMAGE_ROLE, 4, Some(&BTreeSet::new()), None)
        .expect("planned target embedding and head");
    (drafter, target)
}

/// Deterministic tap rows: tap t's values lie in (-(t + 1), t + 1).
fn taps() -> Vec<Vec<u16>> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    (0..5)
        .map(|tap| {
            (0..CONTEXT * HIDDEN)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
                    f32_to_bf16((2.0 * unit - 1.0) * (tap + 1) as f32)
                })
                .collect()
        })
        .collect()
}

fn phase_inputs(
    session: &Session,
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    tokens: &[i32],
    start: usize,
) -> LabResult<DeviceInputs> {
    let rows = tokens.len();
    let mut item = |values: &[u8]| allocate(buffers, session, TOKENS * 4, values);
    let positions = (start..start + rows).map(|p| p as i32).collect::<Vec<_>>();
    let zero = item(&[])?;
    Ok(DeviceInputs {
        token_ids: item(&i32_bytes(tokens))?,
        positions: item(&i32_bytes(&positions))?,
        token_batch: zero,
        block_tables: zero,
        slot_ids: zero,
        q_start: zero,
        q_len: item(&i32_bytes(&[rows as i32]))?,
        prefix_len: item(&i32_bytes(&[start as i32]))?,
        seq_lens: item(&i32_bytes(&[(start + rows) as i32]))?,
        live_batch: item(&1_u32.to_le_bytes())?,
        last_positions: item(&((start + rows - 1) as i64).to_le_bytes())?,
        live_tokens: item(&(rows as u32).to_le_bytes())?,
        logit_rows: zero,
        row_indices: zero,
    })
}

/// What every rank's stage shares: the projection choices, the prepared ops
/// and the switches they were prepared with.
struct Shared {
    choices: Choices,
    ops: Ops,
    switches: Switches,
}

/// The component a DFlash package of `checkpoint` carries, authenticated,
/// and what every rank's stage shares.
fn authenticated(checkpoint: &str) -> (bundle_runtime::projection::Authenticated, Shared) {
    let tp = Tp::TP4;
    let components = precision_components(Path::new(checkpoint), Some(DrafterKind::Dflash))
        .expect("precision components");
    let switches = Switches::configured();
    let target = sm120::own_ops::Sm120Own::new(tp);
    let authenticated =
        Prepared::authenticate(&components[&component_id(tp)], tp, switches, &target)
            .expect("authenticated precision component");
    let shared = Shared {
        choices: Choices::new(
            authenticated.document(),
            |id, usage| authenticated.locate(id, usage),
            switches,
            &target,
        )
        .expect("projection choices"),
        ops: Ops::prepare(
            &*sm120::backend(tp, switches, true).expect("SM120 backend"),
            &Plan {
                tp,
                switches,
                kv: kernel_core::contract::DType::Bf16,
                items: DECODE_ENTRY_ROWS,
                target: &target,
                state: &|_| [u64::MAX; 3],
                gathers: &|_| GatherRegions {
                    rows: [u64::MAX; 2],
                    exchange: Some([u64::MAX; 2]),
                },
                dflash: Some([u64::MAX; 2]),
                own: OwnPlan::new(&target, DECODE_ENTRY_ROWS),
            },
        )
        .expect("prepared ops"),
        switches,
    };
    (authenticated, shared)
}

/// The index in `COMM_BUCKETS` of `phase`'s rows.
fn comm_bucket(phase: Phase) -> usize {
    COMM_BUCKETS
        .iter()
        .position(|&rows| Some(rows) == phase.capacity_in::<Mimo>())
        .expect("a comm bucket of the phase's rows")
}

fn rank(
    session: &Session,
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    (drafter, target): (&PlannedImages, &PlannedImages),
    authenticated: &bundle_runtime::projection::Authenticated,
    ordinal: u32,
) -> LabResult<Rank> {
    let rank = ordinal as usize;
    let index = bundle_mimo26::index::parse(&drafter.index_bytes[rank], ordinal)?;
    let bytes = drafter.read_rank(rank)?;
    let image = allocate(buffers, session, bytes.len(), &bytes)?;
    let mut shared = BTreeMap::new();
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        let data = target.read_named(rank, name)?;
        shared.insert(
            name.to_owned(),
            allocate(buffers, session, data.len(), &data)?,
        );
    }
    let views =
        bundle_mimo26::dflash::views::spans(&index, image, &shared, bundle_mimo26::index::Tp::TP4)
            .map_err(str::to_owned)?;
    let vocab_width = allocate(buffers, session, 4, &(VOCAB_PER_RANK as u32).to_le_bytes())?;
    let rope_tables = bundle_launch::hf_rope_tables(buffers, session, TOKENS)?;
    let mut alloc = |bytes: usize| allocate(buffers, session, bytes, &[]);
    let hidden = TOKENS * HIDDEN * 2;
    let dummy = alloc(4096)?;
    let comm = vec![
        CommState {
            hidden_flags: alloc(4 * 4 * TOKENS * 4)?,
            hidden_epoch: alloc(TOKENS * 4)?,
            gather_rows_flags: dummy,
            gather_rows_epoch: dummy,
            exchange_rows: None,
        },
        CommState {
            hidden_flags: alloc(4 * 4 * BLOCK * 4)?,
            hidden_epoch: alloc(BLOCK * 4)?,
            gather_rows_flags: dummy,
            gather_rows_epoch: dummy,
            exchange_rows: None,
        },
    ];
    let scratch = RankScratch {
        rope_tables: Some(rope_tables),
        buffers: ScratchBuffers {
            qkv: alloc(TOKENS * 3712 * 2)?,
            q: alloc(TOKENS * 16 * 192 * 2)?,
            k_new: alloc(TOKENS * 2 * 192 * 2)?,
            v_new: alloc(TOKENS * 2 * 128 * 2)?,
            attention: alloc(TOKENS * 16 * 128 * 2)?,
            hidden_partial: alloc(hidden)?,
            dense_gate: alloc(hidden)?,
            router_logits: dummy,
            topk_ids: dummy,
            topk_weights: dummy,
            expert_counts: dummy,
            expert_offsets: dummy,
            sorted_rows: dummy,
            padded_rows: dummy,
            gathered: dummy,
            routed_gate: dummy,
            routed_down: dummy,
            combined: dummy,
            comm_inbox: alloc(4 * hidden)?,
            comm_state: comm.clone(),
            active_experts: dummy,
            all_experts: dummy,
            unit_scale: dummy,
            gemm_semaphores: alloc(TOKENS * HIDDEN / 128 * 4)?,
            attention_fold_counters: dummy,
            route_ticket: alloc(
                kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES as usize,
            )?,
        },
    };
    let dflash = RankBuffers {
        views,
        cache: alloc(program::cache_bytes(bundle_mimo26::topology::Tp::TP4, 32) as usize)?,
        taps: alloc(program::tap_bytes(bundle_mimo26::topology::Tp::TP4) as usize)?,
        pairs: alloc(program::PAIR_BYTES as usize)?,
        gathered: alloc(4 * program::PAIR_BYTES as usize)?,
        gather_flags: alloc(program::gather_flag_bytes(bundle_mimo26::topology::Tp::TP4) as usize)?,
        gather_epoch: alloc(program::GATHER_EPOCH_BYTES as usize)?,
        vocab_width,
    };
    let activation = alloc(hidden)?;
    let normalized = alloc(hidden)?;
    let workspace = alloc(
        bundle_mimo26::sm120::launch::workspace_bytes(
            bundle_mimo26::topology::Tp::TP4,
            bundle_runtime::geometry::Items::new(bundle_mimo26::contract::DECODE_ENTRY_ROWS)
                .unwrap(),
        )
        .map_err(|e| e.0)? as usize,
    )?;
    let logits = alloc(TIMED_ROWS * VOCAB_PER_RANK * 2)?;
    let mut block = vec![BONUS as i32];
    block.resize(BLOCK, 0);
    let inputs = Phases {
        context: phase_inputs(session, buffers, &[0; CONTEXT], 0)?,
        verify: phase_inputs(session, buffers, &block, CONTEXT)?,
    };
    let store = rank_store(authenticated, Tp::TP4, &dflash.views, |id| {
        id.starts_with("dflash.") || id.starts_with("head.")
    })?;
    Ok(Rank {
        dflash,
        store,
        scratch,
        comm,
        activation,
        normalized,
        workspace,
        logits,
        inputs,
    })
}

fn bound<'a>(ranks: &'a [Rank], shared: &'a Shared, rank: usize, verify: bool) -> RankStage<'a> {
    let data = &ranks[rank];
    let comm = usize::from(verify);
    let phase = if verify {
        Phase::Verify(BLOCK as u32)
    } else {
        Phase::Prefill
    };
    RankStage {
        services: None,
        projections: Some(Projections {
            store: &data.store,
            choices: &shared.choices,
        }),
        tp: bundle_mimo26::topology::Tp::TP4,
        items: bundle_runtime::geometry::Items::new(bundle_mimo26::contract::DECODE_ENTRY_ROWS)
            .unwrap(),
        ring: bundle_mimo26::ring::RingGeometry::new(bundle_mimo26::topology::Tp::TP4, 32),
        rank: rank as u32,
        phase,
        live_tokens: if verify { BLOCK } else { CONTEXT } as u32,
        live_batch: 1,
        context: (CONTEXT + if verify { BLOCK } else { 0 }) as u32,
        ga_kv: kernel_core::contract::DType::Bf16,
        ga_kv_scales: &[],
        switches: shared.switches,
        ops: &shared.ops,
        bucket: comm_bucket(phase),
        views: &data.dflash.views,
        inputs: if verify {
            data.inputs.verify
        } else {
            data.inputs.context
        },
        scratch: &data.scratch,
        activation: data.activation,
        normalized: data.normalized,
        workspace: data.workspace,
        local_logits: data.logits,
        scored_prompt_logits: false,
        logit_rows: &[],
        logits_gather: LogitsBuffers {
            local: data.logits,
            shard: bundle_mimo26::topology::Tp::TP4.vocab() as u32,
            outputs: vec![data.logits; 4],
            flags: vec![data.logits; 4],
            epochs: vec![data.logits; 4],
        },
        exchange_logits: false,
        hidden_inboxes: ranks.iter().map(|peer| peer.scratch.comm_inbox).collect(),
        hidden_flags: ranks
            .iter()
            .map(|peer| peer.comm[comm].hidden_flags)
            .collect(),
        hidden_epochs: ranks
            .iter()
            .map(|peer| peer.comm[comm].hidden_epoch)
            .collect(),
        hidden_gathered: ranks.iter().map(|peer| peer.normalized).collect(),
        ring_state: data.workspace,
        kv_pool: data.workspace,
    }
}

fn cosine(actual: &[f64], reference: &[f64]) -> f64 {
    let dot = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
    dot / (norm(actual) * norm(reference))
}

fn max_error(actual: &[f64], reference: &[f64]) -> f64 {
    actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Index of the largest value, the lowest on ties (+0 equal to -0), as
/// torch.argmax picks.
fn argmax(values: &[f64]) -> LabResult<usize> {
    let best =
        topk(values, 1, Ties::IeeeLowerIndex, NanKeys::Refuse, None).map_err(|e| e.0.to_owned())?;
    Ok(best[0])
}

fn bf16_values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|v| f64::from(bf16_to_f32(u16::from_le_bytes([v[0], v[1]]))))
        .collect()
}

#[test]
#[ignore = "requires four GPUs and the MiMo checkpoint"]
fn dflash_drafter_matches_f64_oracle() {
    let checkpoint = std::env::var("CHOPINFER_MIMO_CHECKPOINT").expect("checkpoint path");
    let (drafter, target) = planned_images(&checkpoint);
    let (authenticated, shared) = authenticated(&checkpoint);
    let receipt = std::env::var("CHOPINFER_MIMO_DFLASH_RECEIPT").expect("receipt path");
    assert_eq!(
        kernel_lab::selected_ordinals()
            .unwrap()
            .expect("GPU ordinals")
            .len(),
        4
    );
    kernel_lab::gpu_test_multi(6 << 30, |sessions| {
        let mut buffers = BTreeMap::new();
        let ranks = (0..4)
            .map(|r| {
                rank(
                    &sessions[r],
                    &mut buffers,
                    (&drafter, &target),
                    &authenticated,
                    r as u32,
                )
            })
            .collect::<LabResult<Vec<_>>>()?;
        let peers = Peers {
            gathered: ranks.iter().map(|peer| peer.dflash.gathered).collect(),
            gather_flags: ranks.iter().map(|peer| peer.dflash.gather_flags).collect(),
            gather_epochs: ranks.iter().map(|peer| peer.dflash.gather_epoch).collect(),
        };
        let refuse = |e: kernel_core::contract::Refusal| e.0.to_owned();
        let plans = (0..4)
            .map(|r| {
                let context = bound(&ranks, &shared, r, false);
                let taps = (0..5)
                    .map(|tap| program::tap(&context, &ranks[r].dflash, tap, 0))
                    .map(|lowered| lowered.map(program_lower::launches))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(refuse)?;
                let projection = program::context(&context, &ranks[r].dflash, None)
                    .map(program_lower::launches)
                    .map_err(refuse)?;
                let block =
                    program::block(&bound(&ranks, &shared, r, true), &ranks[r].dflash, &peers)
                        .map(program_lower::launches)
                        .map_err(refuse)?;
                Ok((taps, projection, block))
            })
            .collect::<LabResult<Vec<_>>>()?;
        for (rank, (_, context, block)) in ranks.iter().zip(&plans) {
            let ropes: Vec<_> = context
                .iter()
                .chain(block)
                .filter(|p| p.symbol.contains("rope"))
                .collect();
            assert!(ropes.iter().any(|p| p.symbol.contains("q_rope_table")));
            assert!(ropes.iter().any(|p| p.symbol.contains("k_rope_table")));
            for plan in ropes {
                bundle_launch::assert_hf_table_plan(plan, rank.scratch.rope_tables.unwrap());
            }
        }
        let needed = plans
            .iter()
            .flat_map(|(taps, projection, block)| {
                taps.iter().flatten().chain(projection).chain(block)
            })
            .map(|plan: &PreparedLaunch| plan.symbol.as_str())
            .collect::<BTreeSet<_>>();
        let functions = bundle_launch::functions(sessions, &needed)?;
        let taps = taps();
        for (tap, rows) in taps.iter().enumerate() {
            for (r, session) in sessions.iter().enumerate() {
                write(&buffers, session, ranks[r].activation, &bf16_bytes(rows))?;
                for plan in &plans[r].0[tap] {
                    launch(session, &buffers, &functions[r], plan)?;
                }
            }
        }
        // Each rank's collectives wait for the others on the device, so every
        // rank's whole list is queued before any result is read.
        for (r, session) in sessions.iter().enumerate() {
            for plan in plans[r].1.iter().chain(&plans[r].2) {
                launch(session, &buffers, &functions[r], plan)?;
            }
        }
        let mut gpu_logits = vec![Vec::new(); BLOCK];
        let mut drafted = Vec::new();
        let mut gpu_cache = Vec::new();
        for (r, session) in sessions.iter().enumerate() {
            let local = bf16_values(&read(
                &buffers,
                session,
                ranks[r].logits,
                BLOCK * VOCAB_PER_RANK * 2,
            )?);
            for (row, values) in gpu_logits.iter_mut().enumerate() {
                values.extend_from_slice(&local[row * VOCAB_PER_RANK..][..VOCAB_PER_RANK]);
            }
            let tokens = read(
                &buffers,
                session,
                ranks[r].inputs.verify.token_ids,
                BLOCK * 4,
            )?;
            drafted.push(
                tokens
                    .chunks_exact(4)
                    .map(|v| u32::from_le_bytes(v.try_into().expect("four bytes")))
                    .collect::<Vec<_>>(),
            );
            let mut planes = Vec::new();
            for layer in [0, 4] {
                let (k, v) = program::cache_planes(
                    ranks[r].dflash.cache,
                    layer,
                    bundle_mimo26::topology::Tp::TP4,
                    32,
                )
                .map_err(refuse)?;
                planes.push((
                    bf16_values(&read(&buffers, session, k, CACHE * 256 * 2)?),
                    bf16_values(&read(&buffers, session, v, CACHE * 256 * 2)?),
                ));
            }
            gpu_cache.push(planes);
        }

        let oracle = DflashOracle::load(Path::new(&checkpoint)).map_err(|e| e.to_string())?;
        let features = (0..CONTEXT)
            .flat_map(|row| {
                taps.iter()
                    .flat_map(move |tap| &tap[row * HIDDEN..][..HIDDEN])
            })
            .copied()
            .collect::<Vec<_>>();
        let context = oracle
            .context(&features, &(0..CONTEXT).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?;
        let ends = ModelEnds::load(Path::new(&checkpoint)).map_err(|e| e.to_string())?;
        let mut embeddings = ends.embed(&[BONUS]).map_err(|e| e.to_string())?;
        for _ in 1..BLOCK {
            embeddings.extend_from_slice(oracle.mask_embedding());
        }
        let normalized = oracle
            .block(&embeddings, CONTEXT, &context)
            .map_err(|e| e.to_string())?;
        let reference = ends.logits(&normalized);

        let mut failures = Vec::new();
        let mut cache_results = Vec::new();
        for (r, planes) in gpu_cache.iter().enumerate() {
            for ((k, v), layer) in planes.iter().zip([0_usize, 4]) {
                // The cache keeps the last 2048 positions at position % 2048;
                // rank r holds KV heads 2r and 2r + 1.
                let kept = CONTEXT.saturating_sub(CACHE)..CONTEXT;
                let select = |values: &[f64]| {
                    kept.clone()
                        .flat_map(|p| values[(p * 8 + 2 * r) * 128..][..256].to_vec())
                        .collect::<Vec<_>>()
                };
                let cached = |values: &[f64]| {
                    kept.clone()
                        .flat_map(|p| values[p % CACHE * 256..][..256].to_vec())
                        .collect::<Vec<_>>()
                };
                let (keys, values) = (select(&context[layer].keys), select(&context[layer].values));
                let (k, v) = (&cached(k), &cached(v));
                let result = serde_json::json!({"rank": r, "layer": layer,
                    "k_cosine": cosine(k, &keys), "k_max_abs_error": max_error(k, &keys),
                    "v_cosine": cosine(v, &values), "v_max_abs_error": max_error(v, &values)});
                if cosine(k, &keys) < 0.9999 || cosine(v, &values) < 0.9999 {
                    failures.push(format!("context cache {result}"));
                }
                cache_results.push(result);
            }
        }
        let mut row_results = Vec::new();
        for row in 1..BLOCK {
            let actual = &gpu_logits[row];
            let expected = &reference[row];
            let top = argmax(actual)?;
            let oracle_top = argmax(expected)?;
            let best =
                topk(expected, 2, Ties::LowerIndex, NanKeys::Refuse, None).map_err(refuse)?;
            let margin = expected[best[0]] - expected[best[1]];
            let cosine = cosine(actual, expected);
            let result = serde_json::json!({"row": row, "cosine": cosine,
                "max_abs_error": max_error(actual, expected), "gpu_top": top,
                "oracle_top": oracle_top, "oracle_margin": margin,
                "drafts": drafted.iter().map(|tokens| tokens[row]).collect::<Vec<_>>()});
            if cosine < 0.999 || (top != oracle_top && margin > 0.1) {
                failures.push(format!("draft logits {result}"));
            }
            if drafted.iter().any(|tokens| tokens[row] as usize != top) {
                failures.push(format!("drafted token differs from GPU argmax {result}"));
            }
            row_results.push(result);
        }
        if drafted.iter().any(|tokens| tokens[0] != BONUS) {
            failures.push("the bonus token changed".into());
        }
        std::fs::write(
            &receipt,
            serde_json::to_vec_pretty(&serde_json::json!({
                "gate": "cache cosine >= 0.9999; draft logits cosine >= 0.999; \
                         top-1 equals the oracle's unless its margin is <= 0.1; \
                         drafts equal the GPU argmax on every rank",
                "context_rows": CONTEXT, "block_rows": BLOCK, "bonus": BONUS,
                "cache": cache_results, "rows": row_results, "failures": failures,
                "devices": sessions.iter().map(|s| &s.metadata).collect::<Vec<_>>()}))
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("DFlash drafter GPU oracle gate");
}

/// Inputs of `items` items of `rows` rows each, every item in its own cache
/// slot with `prefix` committed positions before its first row.
fn batch_inputs(
    session: &Session,
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    items: usize,
    rows: usize,
    prefix: usize,
) -> LabResult<DeviceInputs> {
    let total = items * rows;
    let mut item = |values: &[u8]| allocate(buffers, session, TOKENS * 4, values);
    let per_item =
        |value: &dyn Fn(usize) -> i32| i32_bytes(&(0..items).map(value).collect::<Vec<_>>());
    let tokens = (0..total)
        .map(|row| if row % rows == 0 { BONUS as i32 } else { 0 })
        .collect::<Vec<_>>();
    let positions = (0..total)
        .map(|row| (prefix + row % rows) as i32)
        .collect::<Vec<_>>();
    let batch = (0..total)
        .map(|row| (row / rows) as i32)
        .collect::<Vec<_>>();
    let last = (0..items)
        .flat_map(|_| ((prefix + rows - 1) as i64).to_le_bytes())
        .collect::<Vec<_>>();
    let zero = item(&[])?;
    Ok(DeviceInputs {
        token_ids: item(&i32_bytes(&tokens))?,
        positions: item(&i32_bytes(&positions))?,
        token_batch: item(&i32_bytes(&batch))?,
        block_tables: zero,
        slot_ids: item(&per_item(&|i| i as i32))?,
        q_start: item(&per_item(&|i| (i * rows) as i32))?,
        q_len: item(&per_item(&|_| rows as i32))?,
        prefix_len: item(&per_item(&|_| prefix as i32))?,
        seq_lens: item(&per_item(&|_| (prefix + rows) as i32))?,
        live_batch: item(&(items as u32).to_le_bytes())?,
        last_positions: item(&last)?,
        live_tokens: item(&(total as u32).to_le_bytes())?,
        logit_rows: zero,
        row_indices: zero,
    })
}

/// A decode or verify stage over `inputs`, using hidden comm state `comm`.
fn timed_bound<'a>(
    ranks: &'a [Rank],
    shared: &'a Shared,
    rank: usize,
    phase: Phase,
    inputs: DeviceInputs,
    comm: usize,
    (items, rows, prefix): (usize, usize, usize),
) -> RankStage<'a> {
    RankStage {
        phase,
        bucket: comm_bucket(phase),
        live_tokens: (items * rows) as u32,
        live_batch: items as u32,
        context: (prefix + rows) as u32,
        inputs,
        hidden_flags: ranks
            .iter()
            .map(|peer| peer.comm[comm].hidden_flags)
            .collect(),
        hidden_epochs: ranks
            .iter()
            .map(|peer| peer.comm[comm].hidden_epoch)
            .collect(),
        ..bound(ranks, shared, rank, true)
    }
}

#[test]
#[ignore = "benchmark: requires four GPUs and the MiMo checkpoint"]
fn dflash_drafter_step_cost() {
    let checkpoint = std::env::var("CHOPINFER_MIMO_CHECKPOINT").expect("checkpoint path");
    let (drafter, target) = planned_images(&checkpoint);
    let (authenticated, shared) = authenticated(&checkpoint);
    // (items, rows per item); one-row items are decode steps that only write
    // context, the others verify steps that also draft.
    const BATCHES: [(usize, usize); 8] = [
        (1, 1),
        (4, 1),
        (16, 1),
        (32, 1),
        (1, 8),
        (4, 8),
        (8, 4),
        (16, 2),
    ];
    // A full 1023-position window behind every query.
    const PREFIX: usize = 1100;
    assert_eq!(
        kernel_lab::selected_ordinals()
            .unwrap()
            .expect("GPU ordinals")
            .len(),
        4
    );
    kernel_lab::gpu_test_multi(6 << 30, |sessions| {
        let mut buffers = BTreeMap::new();
        let mut ranks = (0..4)
            .map(|r| {
                rank(
                    &sessions[r],
                    &mut buffers,
                    (&drafter, &target),
                    &authenticated,
                    r as u32,
                )
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut inputs = vec![Vec::new(); 4];
        for (r, session) in sessions.iter().enumerate() {
            for &(items, rows) in &BATCHES {
                let capacity = TIMED_ROWS;
                let flags = allocate(&mut buffers, session, 4 * 4 * capacity * 4, &[])?;
                let epoch = allocate(&mut buffers, session, capacity * 4, &[])?;
                let dummy = ranks[r].comm[0].gather_rows_flags;
                ranks[r].comm.push(CommState {
                    hidden_flags: flags,
                    hidden_epoch: epoch,
                    gather_rows_flags: dummy,
                    gather_rows_epoch: dummy,
                    exchange_rows: None,
                });
                inputs[r].push(batch_inputs(session, &mut buffers, items, rows, PREFIX)?);
            }
        }
        let peers = Peers {
            gathered: ranks.iter().map(|peer| peer.dflash.gathered).collect(),
            gather_flags: ranks.iter().map(|peer| peer.dflash.gather_flags).collect(),
            gather_epochs: ranks.iter().map(|peer| peer.dflash.gather_epoch).collect(),
        };
        let refuse = |e: kernel_core::contract::Refusal| e.0.to_owned();
        let mut results = Vec::new();
        for (index, &(items, rows)) in BATCHES.iter().enumerate() {
            let verify = rows > 1;
            let phase = if verify {
                Phase::Verify((items * rows) as u32)
            } else {
                Phase::Decode(items as u32)
            };
            // [taps and context, block] per rank.
            let plans = (0..4)
                .map(|r| {
                    let stage = timed_bound(
                        &ranks,
                        &shared,
                        r,
                        phase,
                        inputs[r][index],
                        2 + index,
                        (items, rows, PREFIX),
                    );
                    let mut context = Vec::new();
                    for tap in 0..5 {
                        context.extend(
                            program::tap(&stage, &ranks[r].dflash, tap, 0).map_err(refuse)?,
                        );
                    }
                    context
                        .extend(program::context(&stage, &ranks[r].dflash, None).map_err(refuse)?);
                    let block = if verify {
                        program::block(&stage, &ranks[r].dflash, &peers).map_err(refuse)?
                    } else {
                        Vec::new()
                    };
                    Ok([
                        program_lower::launches(context),
                        program_lower::launches(block),
                    ])
                })
                .collect::<LabResult<Vec<_>>>()?;
            for (rank, stages) in ranks.iter().zip(&plans) {
                let rope = stages
                    .iter()
                    .flatten()
                    .filter(|plan| plan.symbol.contains("rope"));
                let mut count = 0;
                for plan in rope {
                    bundle_launch::assert_hf_table_plan(plan, rank.scratch.rope_tables.unwrap());
                    count += 1;
                }
                assert!(count > 0, "streaming context must rotate keys");
            }
            let needed = plans
                .iter()
                .flatten()
                .flatten()
                .map(|plan: &PreparedLaunch| plan.symbol.as_str())
                .collect::<BTreeSet<_>>();
            let functions = bundle_launch::functions(sessions, &needed)?;
            let mut row = serde_json::json!({"items": items, "rows_per_item": rows,
                "rows": items * rows, "verify": verify, "prefix": PREFIX});
            let parts = if verify { 3 } else { 1 };
            for part in 0..parts {
                // Part 0 is the context work, 1 the block and 2 both.
                let graphs = sessions
                    .iter()
                    .enumerate()
                    .map(|(r, session)| {
                        session.capture(|_| {
                            let selected = match part {
                                0 => vec![&plans[r][0]],
                                1 => vec![&plans[r][1]],
                                _ => vec![&plans[r][1], &plans[r][0]],
                            };
                            for plan in selected.into_iter().flatten() {
                                launch(session, &buffers, &functions[r], plan)?;
                            }
                            Ok(())
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let timing = kernel_lab::time_multi_graph(sessions, &graphs, 1, 5, 30)?;
                let name = ["context_ms", "block_ms", "step_ms"][part];
                row[name] = serde_json::json!(timing.critical_path.median_ms);
                row[name.replace("_ms", "_min_ms")] =
                    serde_json::json!(timing.critical_path.min_ms);
            }
            eprintln!("{row}");
            results.push(row);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("batches".into(), serde_json::json!(results));
        receipt.insert(
            "ordinals".into(),
            serde_json::json!(sessions.iter().map(|s| s.ordinal).collect::<Vec<_>>()),
        );
        sessions[0].receipt("dflash_drafter_step_cost", receipt)?;
        Ok(())
    })
    .expect("DFlash drafter step cost");
}
