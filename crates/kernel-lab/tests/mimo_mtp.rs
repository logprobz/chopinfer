//! Execute the bundle's MTP layer 0 plans against the real-weight f64 oracle.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[path = "common/bundle_launch.rs"]
mod bundle_launch;
use bundle_launch::{allocate, bf16_bytes, launch};

#[path = "common/mimo_planned.rs"]
mod mimo_planned;
use mimo_planned::PlannedImages;
use model_mimo26::catalog::{DrafterKind, precision_components};
use model_mimo26_contract::{
    prepared::component_id,
    resources::ScratchBuffers,
    weight_plan::{DRAFTER_IMAGE_ROLE, Drafter},
};

use bundle_mimo26::{
    contract::{DECODE_ENTRY_ROWS, Mimo},
    mtp,
    ops::{GatherRegions, LogitsBuffers, Ops, OwnPlan, Plan},
    program_lower::{DeviceInputs, RankStage, launches},
    projection::{Choices, Prepared, Rank as Projections, rank_store},
    scratch::{COMM_BUCKETS, CommState, RankScratch},
    selection::Phase,
    sm120,
    switches::Switches,
    topology::Tp,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_lab::{LabResult, Session};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;
use runtime_cuda::DeviceBuffer;

struct Rank {
    views: BTreeMap<String, Span>,
    scratch: RankScratch,
    activation: Span,
    normalized: Span,
    hidden: Span,
    embedding: Span,
    packed: Span,
    workspace: Span,
    ring: Span,
    inputs: DeviceInputs,
}

fn rank(
    session: &Session,
    buffers: &mut BTreeMap<u32, DeviceBuffer>,
    images: &PlannedImages,
    ordinal: u32,
) -> LabResult<Rank> {
    let index = bundle_mimo26::index::parse(&images.index_bytes[ordinal as usize], ordinal)?;
    mtp::views::validate(&index, bundle_mimo26::index::Tp::TP4).map_err(str::to_owned)?;
    let bytes = images.read_rank(ordinal as usize)?;
    let image = allocate(buffers, session, bytes.len(), &bytes)?;
    let rope_tables = bundle_launch::hf_rope_tables(buffers, session, 128)?;
    let mut alloc = |bytes| allocate(buffers, session, bytes, &[]);
    let activation = alloc(8192)?;
    let normalized = alloc(8192)?;
    let hidden = alloc(8192)?;
    let embedding = alloc(8192)?;
    let packed = alloc(4096)?;
    let workspace = alloc(64 << 20)?;
    let ring = alloc(32 * 128 * 1280)?;
    let dummy = alloc(4096)?;
    let state = CommState {
        hidden_flags: alloc(64)?,
        hidden_epoch: alloc(4)?,
        gather_rows_flags: dummy,
        gather_rows_epoch: dummy,
        exchange_rows: None,
    };
    let scratch = RankScratch {
        rope_tables: Some(rope_tables),
        buffers: ScratchBuffers {
            qkv: alloc(3712 * 2)?,
            q: alloc(16 * 192 * 2)?,
            k_new: alloc(2 * 192 * 2)?,
            v_new: alloc(2 * 128 * 2)?,
            attention: alloc(16 * 128 * 2)?,
            hidden_partial: alloc(8192)?,
            dense_gate: alloc(8192)?,
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
            // The collective uses fixed protocol regions, even for one row.
            comm_inbox: alloc(4 * 8192 * 8192)?,
            comm_state: vec![state],
            active_experts: dummy,
            all_experts: dummy,
            unit_scale: dummy,
            gemm_semaphores: alloc(4096)?,
            attention_fold_counters: dummy,
            route_ticket: alloc(
                kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES as usize,
            )?,
        },
    };
    let zero = allocate(buffers, session, 128, &[])?;
    let one = allocate(buffers, session, 128, &1_u32.to_le_bytes())?;
    let positions = allocate(buffers, session, 128, &[])?;
    let last_positions = allocate(buffers, session, 128, &[])?;
    let seq_lens = allocate(buffers, session, 128, &1_u32.to_le_bytes())?;
    let inputs = DeviceInputs {
        token_ids: zero,
        positions,
        token_batch: zero,
        block_tables: zero,
        slot_ids: zero,
        q_start: zero,
        q_len: one,
        prefix_len: positions,
        seq_lens,
        live_batch: one,
        last_positions,
        live_tokens: one,
        logit_rows: one,
        row_indices: zero,
    };
    let target = BTreeMap::from([
        ("model.embed_tokens.weight".into(), embedding),
        ("lm_head.weight".into(), dummy),
    ]);
    let views = mtp::views::aliases(&index, image, &target, bundle_mimo26::index::Tp::TP4)
        .map_err(str::to_owned)?;
    Ok(Rank {
        views,
        scratch,
        activation,
        normalized,
        hidden,
        embedding,
        packed,
        workspace,
        ring,
        inputs,
    })
}

/// What every rank's stage shares: the projection choices, the prepared ops
/// and the switches they were prepared with.
struct Shared {
    choices: Choices,
    ops: Ops,
    switches: Switches,
}

fn bound<'a>(
    ranks: &'a [Rank],
    rank: usize,
    store: &'a bundle_runtime::projection::Store,
    shared: &'a Shared,
) -> RankStage<'a> {
    let data = &ranks[rank];
    RankStage {
        services: None,
        projections: Some(Projections {
            store,
            choices: &shared.choices,
        }),
        tp: Tp::TP4,
        items: bundle_runtime::geometry::Items::new(bundle_mimo26::contract::DECODE_ENTRY_ROWS)
            .unwrap(),
        ring: bundle_mimo26::ring::RingGeometry::new(Tp::TP4, 32),
        rank: rank as u32,
        phase: Phase::Decode(1),
        live_tokens: 1,
        live_batch: 1,
        context: 2,
        ga_kv: kernel_core::contract::DType::Bf16,
        ga_kv_scales: &[],
        switches: shared.switches,
        ops: &shared.ops,
        bucket: COMM_BUCKETS
            .iter()
            .position(|&rows| Some(rows) == Phase::Decode(1).capacity_in::<Mimo>())
            .expect("the decode bucket of one row"),
        views: &data.views,
        inputs: data.inputs,
        scratch: &data.scratch,
        activation: data.activation,
        normalized: data.normalized,
        workspace: data.workspace,
        local_logits: data.packed,
        scored_prompt_logits: false,
        logit_rows: &[0],
        logits_gather: LogitsBuffers {
            local: data.packed,
            shard: Tp::TP4.vocab() as u32,
            outputs: vec![data.packed; 4],
            flags: vec![data.packed; 4],
            epochs: vec![data.packed; 4],
        },
        exchange_logits: false,
        hidden_inboxes: ranks.iter().map(|peer| peer.scratch.comm_inbox).collect(),
        hidden_flags: ranks
            .iter()
            .map(|peer| peer.scratch.comm_state[0].hidden_flags)
            .collect(),
        hidden_epochs: ranks
            .iter()
            .map(|peer| peer.scratch.comm_state[0].hidden_epoch)
            .collect(),
        hidden_gathered: ranks.iter().map(|peer| peer.normalized).collect(),
        ring_state: data.ring,
        kv_pool: data.ring,
    }
}

#[test]
#[ignore = "requires four idle GPUs and the MiMo checkpoint"]
fn mtp_layer_zero_matches_f64_oracle() {
    let checkpoint = std::env::var("CHOPINFER_MIMO_CHECKPOINT").expect("checkpoint path");
    // The MTP drafter image, planned from the checkpoint.
    let images = PlannedImages::plan(
        Path::new(&checkpoint),
        DRAFTER_IMAGE_ROLE,
        4,
        None,
        Some(Drafter::Mtp),
    )
    .expect("planned MTP image");
    assert_eq!(
        kernel_lab::selected_ordinals()
            .unwrap()
            .expect("GPU ordinals")
            .len(),
        4
    );
    // The MTP projections run the component a served package carries.
    let tp = Tp::TP4;
    let components = precision_components(Path::new(&checkpoint), Some(DrafterKind::Mtp))
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
                dflash: None,
                // The pair packs each rank's EH shard through the own ops.
                own: OwnPlan {
                    mtp: true,
                    ..OwnPlan::new(&target, DECODE_ENTRY_ROWS)
                },
            },
        )
        .expect("prepared ops"),
        switches,
    };
    let receipt = std::env::var("CHOPINFER_MIMO_MTP_RECEIPT").expect("receipt path");
    kernel_lab::gpu_test_multi(2 << 30, |sessions| {
        let mut buffers = BTreeMap::new();
        let ranks = (0..4)
            .map(|r| rank(&sessions[r], &mut buffers, &images, r as u32))
            .collect::<LabResult<Vec<_>>>()?;
        let stores = ranks
            .iter()
            .map(|rank| rank_store(&authenticated, tp, &rank.views, |id| id.starts_with("mtp")))
            .collect::<LabResult<Vec<_>>>()?;
        let plans = (0..4)
            .map(|r| {
                let b = bound(&ranks, r, &stores[r], &shared);
                let weight = ranks[r].views[if r < 2 {
                    "model.mtp.layers.0.enorm.weight"
                } else {
                    "model.mtp.layers.0.hnorm.weight"
                }];
                let mut phases = vec![
                    launches(
                        mtp::program::pair(&b, ranks[r].hidden, weight, ranks[r].packed)
                            .map_err(|e| e.0.to_owned())?,
                    ),
                    vec![
                        mtp::program::eh(&b, ranks[r].packed)
                            .map_err(|e| e.0.to_owned())?
                            .launch,
                    ],
                ];
                for stage in mtp::program::BLOCK {
                    phases.push(b.lower(stage).map_err(|e| e.0.to_owned())?);
                }
                Ok(phases)
            })
            .collect::<LabResult<Vec<_>>>()?;
        for (rank, phases) in ranks.iter().zip(&plans) {
            let ropes: Vec<_> = phases
                .iter()
                .flatten()
                .filter(|p| p.symbol.contains("rope"))
                .collect();
            assert!(!ropes.is_empty());
            for plan in ropes {
                bundle_launch::assert_hf_table_plan(plan, rank.scratch.rope_tables.unwrap());
            }
        }
        let needed = plans
            .iter()
            .flatten()
            .flatten()
            .map(|p| p.symbol.as_str())
            .collect::<BTreeSet<_>>();
        let functions = bundle_launch::functions(sessions, &needed)?;
        let mut oracle = model_mimo26::oracle::mtp::MtpOracle::load(Path::new(&checkpoint))
            .map_err(|e| e.to_string())?;
        let mut history = model_mimo26::oracle::KvHistory::default();
        let mut results = Vec::new();
        for position in 0..2_u32 {
            let hidden = (0..4096)
                .map(|i| f32_to_bf16((((i + position as usize * 17) % 127) as f32 - 63.0) / 64.0))
                .collect::<Vec<_>>();
            let embedding = (0..4096)
                .map(|i| f32_to_bf16((((i + position as usize * 13) % 79) as f32 - 39.0) / 40.0))
                .collect::<Vec<_>>();
            let reference = oracle
                .run(&hidden, &embedding, &mut history, Some(4))
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|v| f64::from(bf16_to_f32(v)))
                .collect::<Vec<_>>();
            for (r, session) in sessions.iter().enumerate() {
                for (s, data) in [
                    (ranks[r].hidden, bf16_bytes(&hidden)),
                    (ranks[r].embedding, bf16_bytes(&embedding)),
                    (ranks[r].inputs.positions, position.to_le_bytes().to_vec()),
                    (
                        ranks[r].inputs.last_positions,
                        u64::from(position).to_le_bytes().to_vec(),
                    ),
                    (
                        ranks[r].inputs.seq_lens,
                        (position + 1).to_le_bytes().to_vec(),
                    ),
                ] {
                    buffers[&s.buffer.slot()]
                        .copy_from_bytes_at(s.offset as usize, &data, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
            }
            for phase in 0..plans[0].len() {
                for (r, session) in sessions.iter().enumerate() {
                    for plan in &plans[r][phase] {
                        launch(session, &buffers, &functions[r], plan)?;
                    }
                    if phase == 1 {
                        buffers[&ranks[r].activation.buffer.slot()]
                            .zero_at_async(0, 8192, &session.stream)
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
            for (r, session) in sessions.iter().enumerate() {
                let actual =
                    session.download_bf16(&buffers[&ranks[r].normalized.buffer.slot()], 4096)?;
                let max_abs_error = model_mimo26::oracle::compare_layer_tokens(&actual, &reference)
                    .map_err(|e| format!("position {position} rank {r}: {e:?}"))?;
                let actual = actual
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
                    .collect::<Vec<_>>();
                let dot = actual
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>().sqrt();
                let cosine = dot / (norm(&actual) * norm(&reference));
                results.push(serde_json::json!({"position": position, "rank": r,
                    "cosine": cosine, "max_abs_error": max_abs_error}));
            }
        }
        std::fs::write(
            &receipt,
            serde_json::to_vec_pretty(
                &serde_json::json!({"gate": "MTP layer 0 cosine >= 0.9999", "results": results,
            "devices": sessions.iter().map(|s| &s.metadata).collect::<Vec<_>>()}),
            )
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    })
    .expect("MTP layer GPU oracle gate");
}
