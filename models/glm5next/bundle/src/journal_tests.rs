//! Fake-host launch journals of the real geometry at TP4 and
//! context-parallel size 1, per latent KV format and phase. Each rank's
//! program goes to the fake device's journal one launch per line: the
//! launch the device receives (symbol, configuration and bound arguments),
//! after the rank's buffer allocations, or for a stage that does not lower
//! (a projection without the precision component) its record: the stage,
//! the op family, the numerical contract, the planner and the query. A
//! pinned digest notices any change in what the program would launch.

use std::collections::{BTreeMap, BTreeSet};

use bundle_runtime::dispatch;
use bundle_runtime::fake_host::{FakeDevice, check_journal};
use bundle_runtime::geometry::Items;
use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{BlockTableMirror, DeviceInputs, InputLayout, PhaseArena};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use model_glm5next_contract::contract::{DECODE_BATCH, Glm, Shape, has_mtp_layer, latent_format};
use model_glm5next_contract::direct::layout;
use model_glm5next_contract::sm120::engines as selected;
use runtime_bundle_abi::{
    BundleDeviceBufferHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1, KernelArgumentV1,
};

use crate::buffers::{CommState, RankBuffers, StatePools, allocate_scratch};
use crate::lower::Lowering;

use crate::launch::{Op, Planner, drafter_records, records};
use crate::program::{DraftPass, Phase, Stage};

pub(crate) const PHASES: [Phase; 3] = [
    Phase::Prefill { rows: 2048 },
    Phase::Decode(8),
    Phase::Verify { rows: 32, items: 8 },
];
pub(crate) const FORMATS: [LatentKvFormat; 2] =
    [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16];

pub(crate) fn planner(planner: Planner) -> String {
    match planner {
        Planner::Projection(id) => format!("projection:{id}"),
        Planner::Backend => "backend".into(),
        Planner::Own => "own".into(),
    }
}

/// The comm state regions of TP4's all-reduces, as an instance's ops keep
/// them (`ops::reduce::StateRegions`), prepared once per test process.
pub(crate) fn comm_regions() -> crate::ops::reduce::StateRegions {
    static REGIONS: std::sync::OnceLock<crate::ops::reduce::StateRegions> =
        std::sync::OnceLock::new();
    *REGIONS.get_or_init(|| {
        let engines = crate::sm120::engines::Engines::shared(
            Shape::TP4,
            &model_glm5next_contract::quantization::FP8_BLOCK128,
        )
        .unwrap();
        let (_, served) = crate::sm120::backend(engines, Shape::TP4);
        crate::ops::reduce::ReduceOps::prepare(&served.1, Shape::TP4, &Default::default())
            .unwrap()
            .regions()
    })
}

/// The comm state regions of each of TP4's drafter argmax pair gathers, as
/// an instance's ops keep them (`ops::draft::DraftOps::regions`).
pub(crate) fn draft_regions() -> [u64; 2] {
    let target = crate::sm120::own_ops::Sm120Own::new(Shape::TP4);
    let (backend, _) = crate::target::Target::backend(
        &target,
        &model_glm5next_contract::quantization::FP8_BLOCK128,
    )
    .unwrap();
    crate::ops::draft::DraftOps::prepare(backend.as_ref(), Shape::TP4, &Default::default())
        .unwrap()
        .regions()
}

/// The fake device's buffers of one rank in create's order: the scratch,
/// then (after the host's per-item state and serving KV pool, 1024 pages)
/// the KDA state pools. The rank's text views lie one after another in one
/// arena.
pub(crate) fn rank_buffers(device: &FakeDevice, rank: u32, format: &KvFormat) -> RankBuffers {
    rank_buffers_serving(device, rank, format, Items::new(DECODE_BATCH).unwrap(), 4)
}

/// [`rank_buffers`] of an instance serving `items` with `verify_rows`
/// verify rows per item (none without speculation, else 2 to
/// `MAX_VERIFY_ROWS`), over the rank image's views where the direct plan
/// places them.
pub(crate) fn rank_buffers_serving(
    device: &FakeDevice,
    rank: u32,
    format: &KvFormat,
    items: Items,
    verify_rows: u32,
) -> RankBuffers {
    let shape = Shape::TP4;
    let handle = BundleDeviceBufferHandleV1::from_parts(1, 1000 + rank);
    let views = layout(shape, rank)
        .unwrap()
        .into_iter()
        .map(|entry| {
            let span = DeviceBufferSpanV1::new(handle, entry.offset, entry.bytes());
            (entry.view.name, span)
        })
        .collect();
    rank_buffers_within(device, rank, format, items, verify_rows, views)
}

/// [`rank_buffers`] with `views`, of an instance serving `items` with
/// `verify_rows` rows per verify item (0 without speculation), whose
/// per-item state sizes by them as create's does.
pub(crate) fn rank_buffers_within(
    device: &FakeDevice,
    rank: u32,
    format: &KvFormat,
    items: Items,
    verify_rows: u32,
    views: BTreeMap<String, DeviceBufferSpanV1>,
) -> RankBuffers {
    let shape = Shape::TP4;
    let handle = |id| BundleDeviceBufferHandleV1::from_parts(1, id);
    let stream = BundleStreamHandleV1::from_parts(1, 3000 + rank);
    let mut allocate = |bytes| {
        let id = device.id();
        Ok::<_, ()>(DeviceBufferSpanV1::new(
            device.allocate(id, stream, bytes, 256),
            0,
            bytes,
        ))
    };
    // The drafter's buffers with the KV formats that hold its layer.
    let scratch = allocate_scratch(shape, has_mtp_layer(format), &mut allocate)
        .unwrap()
        .unwrap();
    let reduced = scratch[&crate::buffers::Buffer::Reduced];
    let comm = CommState::allocate(
        shape,
        has_mtp_layer(format).then(draft_regions),
        reduced,
        comm_regions(),
        &mut allocate,
    )
    .unwrap()
    .unwrap();
    let tail = DeviceBufferSpanV1::new(
        handle(4000 + rank),
        0,
        // The host's per-item state: one slot per item of the largest
        // bucket (`runtime_bundle::state_slots_v1`).
        u64::from(items.capacity()) * shape.indexer_tail_slot_bytes(has_mtp_layer(format)),
    );
    let kv_pool = DeviceBufferSpanV1::new(handle(2000 + rank), 0, 1024 * format.page_bytes);
    let extent = Extent {
        served: items.get(),
        bucket: items.capacity(),
        verify_rows,
    };
    let drafter = has_mtp_layer(format)
        .then(|| crate::buffers::DrafterBuffers::allocate(shape, &mut allocate).unwrap());
    let retained =
        StatePools::retained_bytes(shape, format, extent).map(|bytes| allocate(bytes).unwrap());
    let state = StatePools::new(shape, format, extent, retained, tail, &mut allocate)
        .unwrap()
        .unwrap();
    RankBuffers {
        scratch,
        comm,
        state,
        views,
        kv_pool,
        drafter,
    }
}

/// Rows the head scores: a prefill's 32 items' last rows (each of a
/// shorter prefill's rows), else the phase's rows.
pub(crate) fn head_rows(phase: Phase) -> u32 {
    match phase {
        Phase::Prefill { rows } => DECODE_BATCH.min(rows),
        _ => phase.rows(),
    }
}

/// The rank's fake shared input arena of `phase` for `items`, and its
/// block tables.
fn arena(rank: u32, phase: Phase, items: Items) -> (PhaseArena, InputLayout) {
    let layout = match phase {
        Phase::Prefill { .. } => InputLayout::prefill_largest_within::<Glm>(items),
        Phase::Decode(_) => InputLayout::decode_within(items),
        Phase::Verify { .. } => InputLayout::verify_within::<Glm>(items),
    };
    let handle = |id| BundleDeviceBufferHandleV1::from_parts(1, id);
    let tables = BlockTableMirror::<Glm>::device_bytes(Shape::TP4.model_limits(), items);
    let arena = PhaseArena {
        arena: DeviceBufferSpanV1::new(handle(5000 + rank), 0, layout.end),
        block_tables: DeviceBufferSpanV1::new(handle(6000 + rank), 0, tables),
    };
    (arena, layout)
}

/// The phase's spans in a fake shared input arena of the rank.
pub(crate) fn inputs(rank: u32, phase: Phase) -> DeviceInputs {
    inputs_within(rank, phase, Items::new(DECODE_BATCH).unwrap())
}

/// [`inputs`] of an instance serving `items`.
pub(crate) fn inputs_within(rank: u32, phase: Phase, items: Items) -> DeviceInputs {
    let (arena, layout) = arena(rank, phase, items);
    DeviceInputs::within(arena, &layout)
}

/// Each buffer's bytes by handle slot: the rank's scratch and KDA pools,
/// every rank's all-reduce state, its image arena (where the direct plan ends), the host's tail state, KV
/// pool and input arenas.
pub(crate) fn extents(
    rank: u32,
    buffers: &RankBuffers,
    peers: &[CommState],
    phase: Phase,
) -> BTreeMap<u32, u64> {
    let image = layout(Shape::TP4, rank)
        .unwrap()
        .iter()
        .map(|entry| entry.offset + entry.bytes())
        .max()
        .unwrap();
    let mut extents = BTreeMap::from([(1000 + rank, image)]);
    let tail = &buffers.state.spans;
    let drafter = buffers
        .drafter
        .iter()
        .flat_map(|drafter| drafter.staged.into_iter().chain([drafter.vocab_width]));
    // Every peer's all-reduce state, which the comm entries address.
    let comm = peers.iter().flat_map(CommState::spans);
    for span in buffers
        .scratch
        .values()
        .chain(tail)
        .chain([&buffers.kv_pool])
        .copied()
        .chain(comm)
        .chain(drafter)
    {
        let end = extents.entry(span.buffer.slot()).or_default();
        *end = (*end).max(span.offset + span.length);
    }
    let (arena, _) = arena(rank, phase, Items::new(DECODE_BATCH).unwrap());
    for span in [arena.arena, arena.block_tables] {
        extents.insert(span.buffer.slot(), span.offset + span.length);
    }
    extents
}

/// Every rank's program of `phase` over `kv` through the fake device: the
/// merged engines' stages as the launches the device would receive, the
/// others as their launch records.
fn journal(kv: LatentKvFormat, phase: Phase) -> Vec<String> {
    journal_of(kv, phase, None)
}

/// [`journal`] of the program, or with `pass` of the MTP drafter's over
/// the format whose pages hold its layer.
fn journal_of(kv: LatentKvFormat, phase: Phase, pass: Option<DraftPass>) -> Vec<String> {
    let shape = Shape::TP4;
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| latent_format(format) == kv && has_mtp_layer(format) == pass.is_some())
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let records = match pass {
        Some(pass) => drafter_records(shape, kv, phase, pass).unwrap(),
        None => records(shape, kv, phase).unwrap(),
    };
    let ops = crate::ops::Ops::for_tests(shape, 1);
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<_>>();
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    for (rank, buffers) in ranks.iter().enumerate() {
        let rank = rank as u32;
        let verify_tokens = inputs(rank, PHASES[2]).token_ids;
        let inputs = inputs(rank, phase);
        let extents = extents(rank, buffers, &peers, phase);
        let lowering = Lowering {
            shape,
            format: &format,
            buffers,
            inputs: &inputs,
            ops: &ops,
            rank,
            index_topk: shape.profile.index_topk_max,
            pieces: &[],
            row_offset: 0,
            peers: &peers,
            head_rows: head_rows(phase),
            context_parallel: None,
            draft: pass,
            // The drafter's argmax fills the target's verify tokens.
            verify_tokens: pass.map(|_| verify_tokens),
            projections: None,
        };
        for record in &records {
            let Some(launches) = lowering.lower(phase, record.stage).unwrap() else {
                device.note(format!(
                    "launch r{rank} {:?} rows {} {:?} {} {} {} {:?}",
                    phase,
                    phase.rows(),
                    record.stage,
                    record.op.family(),
                    record.op.contract().unwrap(),
                    planner(record.planner),
                    record.op,
                ));
                continue;
            };
            for launch in launches {
                assert!(
                    crate::cubin_of(&launch.symbol).is_some(),
                    "{} is not embedded",
                    launch.symbol
                );
                let config = dispatch::launch_config(launch.plan).unwrap();
                let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                check_operands(rank, &launch.symbol, &arguments, &extents);
                device.note(format!(
                    "launch r{rank} {:?} {} rows {} {config:?} {arguments:?}",
                    record.stage, launch.symbol, launch.rows,
                ));
            }
        }
    }
    device.journal.take()
}

/// Every span a launch binds lies inside its buffer, and the KDA
/// convolutions read their weights as the plan stores them: F32, `taps`
/// values of each of the rank's channels.
pub(crate) fn check_operands(
    rank: u32,
    symbol: &str,
    arguments: &[KernelArgumentV1],
    extents: &BTreeMap<u32, u64>,
) {
    let p = Shape::TP4.profile;
    for argument in arguments {
        let KernelArgumentV1::DeviceBuffer(span) = argument else {
            continue;
        };
        let extent = extents
            .get(&span.buffer.slot())
            .unwrap_or_else(|| panic!("{symbol}: unknown buffer {span:?}"));
        assert!(
            span.offset + span.length <= *extent,
            "{symbol}: {span:?} past its buffer's {extent} bytes"
        );
        if symbol.contains("_conv_") && span.buffer.slot() == 1000 + rank {
            assert_eq!(
                span.length,
                u64::from(p.linear_heads * p.linear_head_dim * p.conv_taps) * 4,
                "{symbol}: a convolution weight"
            );
        }
    }
}

#[test]
fn every_record_validates_against_its_contract() {
    for kv in FORMATS {
        for phase in PHASES {
            for record in records(Shape::TP4, kv, phase).unwrap() {
                if let Err(refusal) = record.op.contract() {
                    panic!("{phase:?} {:?}: {refusal}", record.stage);
                }
            }
        }
    }
}

#[test]
fn every_op_family_journals_with_its_query() {
    let records = records(
        Shape::TP4,
        LatentKvFormat::Fp8E4M3Block128,
        Phase::Decode(8),
    )
    .unwrap();
    let families: BTreeSet<_> = records.iter().map(|r| r.op.family()).collect();
    assert_eq!(
        families,
        BTreeSet::from([
            "comm",
            "gated-mlp",
            "glm5next-row",
            "gemm",
            "head-batched-gemm",
            "latent-attention",
            "moe",
            "per-channel-decay-delta-rule",
            "pooled-indexer",
            "recurrent",
            "row",
            "sharded-rows",
            "sinkhorn-hc",
        ])
    );
    // The key list holds 2048 selected tokens and the 3-token tail.
    let attention = records
        .iter()
        .find_map(|r| match r.op {
            Op::Latent(LatentAttentionQuery::Attention { max_keys, kv, .. }) => {
                Some((max_keys, kv))
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(attention, (2051, LatentKvFormat::Fp8E4M3Block128));
}

/// Every record names what prepares its launches: the projection API for
/// the projection stages and the head, GLM's own ops for its own rows, the
/// backend's ops for every other stage.
#[test]
fn records_name_what_prepares_their_launches() {
    let mut planners = BTreeSet::new();
    for phase in PHASES {
        let mut programs = vec![records(Shape::TP4, LatentKvFormat::Fp8E4M3Block128, phase)];
        for pass in [DraftPass::Extend, DraftPass::Step] {
            programs.push(drafter_records(
                Shape::TP4,
                LatentKvFormat::Fp8E4M3Block128,
                phase,
                pass,
            ));
        }
        for record in programs.into_iter().flat_map(Result::unwrap) {
            let expected = match (record.stage, record.op) {
                (Stage::Project(_, id), _) => Planner::Projection(id),
                (Stage::LmHead, _) => Planner::Projection("head"),
                (_, Op::Own(_)) => Planner::Own,
                _ => Planner::Backend,
            };
            assert_eq!(record.planner, expected, "{phase:?} {:?}", record.stage);
            planners.insert(
                planner(record.planner)
                    .split(':')
                    .next()
                    .unwrap()
                    .to_owned(),
            );
        }
    }
    assert_eq!(
        planners,
        BTreeSet::from(["backend", "own", "projection"].map(String::from))
    );
}

/// Only the KV formats that hold the drafter's layer allocate its scratch
/// (about 256 MiB per rank), so an instance without it keeps that memory
/// for KV.
#[test]
fn only_drafting_formats_allocate_the_drafter_scratch() {
    let device: FakeDevice = FakeDevice::default();
    for format in Shape::TP4.kv_formats() {
        let buffers = rank_buffers(&device, 0, &format);
        let drafter = crate::buffers::Buffer::ALL
            .iter()
            .filter(|buffer| buffers.scratch.contains_key(buffer) && buffer.drafter())
            .count();
        assert_eq!(drafter, if has_mtp_layer(&format) { 10 } else { 0 });
        // Four rows of the prefill capacity, then its inputs, argmax pairs
        // and shared key lists.
        let bytes: u64 = crate::buffers::Buffer::ALL
            .iter()
            .filter(|buffer| buffer.drafter())
            .map(|buffer| buffer.bytes(Shape::TP4).unwrap())
            .sum();
        assert!((256 << 20..257 << 20).contains(&bytes), "{bytes}");
    }
}

/// The drafter's retained rows are the declared `mtp.retained` state:
/// slot `s`'s `RETAINED_ROWS` rows at entry `s`, back to back, as
/// `crate::drafter` numbers them.
#[test]
fn retained_rows_lie_by_slot_as_the_drafter_numbers_them() {
    let device: FakeDevice = FakeDevice::default();
    let format = Shape::TP4.kv_formats()[1];
    let buffers = rank_buffers(&device, 0, &format);
    let retained = buffers.state.retained().unwrap();
    let row = u64::from(Shape::TP4.profile.hidden) * 2;
    let rows = u64::from(crate::drafter::RETAINED_ROWS);
    // The slots' rows, then the anchor items' spares.
    let anchors = u64::from(model_glm5next_contract::contract::ANCHOR_ITEMS);
    assert_eq!(
        retained.length,
        (u64::from(DECODE_BATCH) + anchors) * rows * row
    );
    let index = buffers
        .state
        .state
        .buffers
        .iter()
        .position(|buffer| buffer.name == "mtp.retained")
        .unwrap();
    for slot in [0, 1, 31] {
        let entry = buffers.state.entry(index, 0, slot);
        assert_eq!(entry.offset, retained.offset + slot * rows * row);
        assert_eq!(entry.length, rows * row);
    }
    let none = rank_buffers(&device, 0, &Shape::TP4.kv_formats()[0]);
    assert!(none.state.retained().is_err());
}

/// Every record the backend's or GLM's own ops prepare lowers to launches
/// of the embedded entries in every phase and format.
#[test]
fn every_backend_and_own_record_lowers() {
    let shape = Shape::TP4;
    let ops = crate::ops::Ops::for_tests(shape, 1);
    let device: FakeDevice = FakeDevice::default();
    for kv in FORMATS {
        let format = shape
            .kv_formats()
            .into_iter()
            .find(|format| latent_format(format) == kv && !has_mtp_layer(format))
            .unwrap();
        let buffers = (0..shape.ranks())
            .map(|rank| rank_buffers(&device, rank, &format))
            .collect::<Vec<_>>();
        let peers = buffers
            .iter()
            .map(|buffers| buffers.comm.clone())
            .collect::<Vec<_>>();
        let buffers = &buffers[0];
        for phase in PHASES {
            let inputs = inputs(0, phase);
            let lowering = Lowering {
                shape,
                format: &format,
                buffers,
                inputs: &inputs,
                ops: &ops,
                rank: 0,
                index_topk: shape.profile.index_topk_max,
                pieces: &[],
                row_offset: 0,
                peers: &peers,
                head_rows: head_rows(phase),
                context_parallel: None,
                draft: None,
                verify_tokens: None,
                projections: None,
            };
            for record in records(shape, kv, phase).unwrap() {
                if !matches!(record.planner, Planner::Projection(_)) {
                    let launches = lowering.lower(phase, record.stage).unwrap();
                    assert!(launches.is_some(), "{:?}", record.stage);
                }
            }
        }
    }
}

// Re-pinned when the KDA checkpoint table took the verify entries' stride
// of 8 IDs per item: each rank's allocation of it is 1,024 bytes (was 768),
// and the verify launches bind its longer span. Re-pinned when the row
// engine lowered the RMSNorms and the embedding gather: their records
// became launches, and each rank allocates the query-down, its norm and
// the raw latent (three more scratch buffers). Re-pinned when the
// all-reduces (F32 one-shot up to 16 rows, else reduce-scatter and
// all-gather, then one rounding), the embedding's widening, the stream
// expansion and the head-row gather lowered: their records became
// launches, each rank allocates the F32 partial, the head rows and its
// all-reduce state, and every rank's buffers are allocated before any
// rank lowers. Re-pinned when the logits moves lowered (decode and verify
// rows exchanged to their owners, prefill rows gathered to rank 0): a new
// stage after the head, whose record now writes F32, and each rank's
// logits shard and full-row buffers with their flags and epochs.
// Re-pinned when the MTP drafter's stream and input buffers joined the
// scratch (three more allocations per rank). Re-pinned when the comm
// flags and epochs were sized for each protocol: the one-shot's for the
// tiles of its most rows, the two-shot's for the phased protocol's fixed
// 82 flag words and one epoch cell (the allocations' lengths and the
// handles after them). Re-pinned when each one-shot row count took its
// own state: 16 states of three allocations per rank (180 more lines).
// Re-pinned when a prefill phase scored up to 64 rows: the logits shard
// and the prefill logits rows hold 64 rows (their lengths and the handles
// after them). Re-pinned when the MoE took the F32 routed sum: its symbol,
// the F32 MLP output (its length and the handles after it), and the F32
// outputs of the experts' down and the shared expert's down records.
// Re-pinned when the drafter's gathered hidden inputs joined the scratch
// (one more allocation per rank). Re-pinned when the drafter's four
// scratch buffers moved to the formats that hold its layer (four fewer
// allocations per rank here).
// Re-pinned when the row-split projections' records wrote F32 partials
// (their output dtype), and when the lowered BF16 projections' weight
// views took their zero rows to the row tile (the views' offsets), and
// when the KDA gates' low-rank buffers joined the scratch (two more
// allocations per rank), and when the router's padded scores took their
// own buffer (one more), and when the latent attention kept every row's 64
// fixed splits (the decode and verify attention's splits and outputs, and
// their merges' splits). Re-pinned when every mixer ran the mix-collapse
// split over CTAs (its symbol, grid and chain sums and tickets, which joined
// the scratch: two more allocations per rank). Re-pinned when prefill
// mixers of 1024 rows and more ran the one-CTA mix-collapse (the prefill
// journals' mixer launches: symbol, grid, no chain sums or tickets).
// Re-pinned when the KDA prefill ran its tiled prepare and scan (their
// symbols and dynamic shared bytes). Re-pinned when the phased all-reduce's
// state joined the comm state (three more allocations per rank, the handles
// after them). Re-pinned when 2 to 32 decode and verify rows reduced
// through the phased entries (each all-reduce and round one launch).
// Re-pinned when the KDA prefill ran its streamed prepare and scan (their
// symbols, the prepare's dynamic shared bytes and the scan's block).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse and
/// its two symbols joined create (the handles).
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies (identity change, accuracy unaffected: their symbols and
/// shape arguments).
#[test]
fn journals_are_pinned_per_kv_format_and_phase() {
    let pinned: [((LatentKvFormat, Phase), (usize, u64)); 6] = [
        ((FORMATS[0], PHASES[0]), (5792, 0x44b9d4094d13023a)),
        ((FORMATS[0], PHASES[1]), (5044, 0xe502cc638f4ef151)),
        ((FORMATS[0], PHASES[2]), (5044, 0x7a52b349bda17d64)),
        ((FORMATS[1], PHASES[0]), (5792, 0x76651e8ef1143ddc)),
        ((FORMATS[1], PHASES[1]), (5044, 0xac9d640261911eaf)),
        ((FORMATS[1], PHASES[2]), (5044, 0x56d5cc0b43d1c0d2)),
    ];
    let mut failures = Vec::new();
    for ((kv, phase), expected) in pinned {
        let lines = journal(kv, phase);
        let file = format!("glm5next-journal-{kv:?}-{}.txt", phase.rows());
        if let Err(error) = check_journal(&lines, expected, &file) {
            failures.push(format!("{kv:?} {phase:?}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The MTP drafter's passes over the FP8 pages that hold its layer: the
/// extend pass over accepted rows (a verify bucket) runs its indexer, a
/// draft step (one row per item) shares the extend pass's selection. Its
/// latent attention, MoE, norms, embedding, all-reduces and residual adds
/// lower; its projections, head, shared selection and argmax are
/// placeholders. Re-pinned when the comm flags and epochs were sized for
/// each protocol, when each one-shot row count took its own state, and
/// when the logits shard held 64 scored rows, when the MoE took the F32
/// routed sum, when the drafter gathered its hidden inputs first (a
/// placeholder) and its argmax stopped gathering logits rows, and when its
/// row inputs, head rows, argmax and shared selection lowered (with their
/// scratch, its retained rows and its candidates' comm state), and when its
/// retained rows became the declared request state (the state pools'
/// allocations), and when the extend pass gathered the target's
/// normalized rows rather than its collapse (the gather's binding), and
/// when the extend pass wrote its indexer tail in place with the prefill
/// writer rather than through the target's verify checkpoints (its key
/// write's symbol, without the checkpoint table), and when the retained
/// rows were allocated before the request state's other pools, as create
/// allocates them before the KV pool (the handles after them).
///
/// Re-pinned when its latent output projection's record wrote an F32
/// partial, and when the lowered BF16 projections' weight views took
/// their zero rows to the row tile (the views' offsets), and when the KDA
/// gates' low-rank buffers joined the scratch (two more allocations per
/// rank), and when its two normalized rows were copied side by side for
/// its input projection (two copies, the norms' outputs and one buffer
/// fewer), and when the router's padded scores took their own buffer (one
/// more allocation per rank), and when the layout copies merged into one
/// entry per layout (the copies' symbols and the handles), and when the
/// latent attention kept every row's 64 fixed splits (the attention's
/// splits and outputs, and the merges' splits), and when every drafter pass
/// took prefill's one split per row (the attention's symbol and splits, no
/// merge), and when the split mix-collapse's chain sums and tickets joined
/// the scratch (two more allocations per rank). Re-pinned when the head's
/// draft rows, their argmax and its candidates' all-reduce took the run's
/// item bucket instead of 32 rows (`program::draft_rows`: the gather and
/// argmax grids and rows, and the all-reduce's elements, 8 items here).
/// Re-pinned when drafter passes took the decode and verify entries' 64
/// fixed splits again (the attention's symbol, grid, splits and workspace
/// binding, and its merge; PERF-B14b). Re-pinned when the argmax moved onto
/// the shared sharded argmax (`model_glm5next_contract::draft_argmax`): its
/// own partial, candidate all-reduce and final became the shared F32
/// partial, the all-gathers of the pairs' scores and ids and the shared
/// final scattered into the drafts and verify tokens (those launches, the
/// pairs' scratch and the gathers' comm state).
#[test]
fn drafter_journals_are_pinned_per_pass() {
    let kv = FORMATS[0];
    let mut failures = Vec::new();
    for (pass, phase, expected) in [
        (
            DraftPass::Extend,
            Phase::Verify { rows: 32, items: 8 },
            (576, 0x727bd523d4167f39),
        ),
        (DraftPass::Step, Phase::Decode(8), (552, 0x4d9bfb042adf4cd2)),
    ] {
        let lines = journal_of(kv, phase, Some(pass));
        // The drafter's layer is the twelfth paged layer: its latent KV
        // write binds the pages from the eleven sparse layers' end.
        let pool = 11 * Shape::TP4.page_layer_bytes(kv);
        assert!(lines.iter().any(|line| {
            line.contains("LatentKvWrite(45)") && line.contains(&format!("offset: {pool}, "))
        }));
        // Its projections and head stay records without the precision
        // component.
        let records = lines
            .iter()
            .filter(|line| line.contains(" projection:"))
            .count();
        assert!(records > 0);
        // Its row inputs and head rows lower in both passes, and its argmax
        // on the shared entries: each rank's partial, the gathers of the
        // pairs' scores and ids, and its final; a draft step compacts the
        // extend pass's key lists instead of running the indexer.
        let launches = |symbol: &str| {
            lines
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains(symbol))
                .count()
        };
        let launched = |symbol: &str| launches(symbol) > 0;
        // The row inputs on GLM's own entry, the head rows on the shared
        // gather by index.
        for symbol in [
            "chop_row_glm5next_draft_gather_v1",
            model_glm5next_contract::sm120::engines::ROW_GATHER_SYMBOL,
        ] {
            assert!(launched(symbol), "{pass:?} {symbol}");
        }
        let ranks = Shape::TP4.ranks() as usize;
        assert!(!launched("chop_row_glm5next_draft_argmax"), "{pass:?}");
        assert_eq!(
            launches(" DraftArgmax chop_row_argmax_partial_f32_v1 "),
            ranks
        );
        assert_eq!(
            launches(" DraftArgmax chop_comm_allgather_f32_v1 "),
            2 * ranks
        );
        assert_eq!(launches(" DraftArgmax chop_row_argmax_final_v1 "), ranks);
        assert_eq!(
            launched("pooled_indexer_glm5next_compact"),
            pass == DraftPass::Step
        );
        assert_eq!(
            launched("pooled_indexer_glm5next_select"),
            pass == DraftPass::Extend
        );
        // Each rank copies its two normalized rows side by side.
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains(" DraftEhInput chop_proj_bf16_copy_"))
                .count(),
            2 * Shape::TP4.ranks() as usize
        );
        let file = format!("glm5next-drafter-{pass:?}-{}.txt", phase.rows());
        if let Err(error) = check_journal(&lines, expected, &file) {
            failures.push(format!("{pass:?}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A word of a fake buffer: its handle slot and byte offset.
type Word = (u32, u64);

/// The `index`th word of `span`, which must hold it.
fn word(span: DeviceBufferSpanV1, index: u64) -> Word {
    assert!((index + 1) * 4 <= span.length, "word {index} past {span:?}");
    (span.buffer.slot(), span.offset + index * 4)
}

/// One step of a CTA of the simulated all-reduces.
#[derive(Clone, Copy, Debug)]
enum CommOp {
    /// Spin until the word holds the value.
    Wait(Word, u32),
    Release(Word, u32),
    /// Add one (a two-shot's arrival).
    Arrive(Word),
}

/// One rank's all-reduce launch: a one-shot's or the low-latency entry's
/// resolved arguments, or the `index`th two-shot collective, simulated as a
/// barrier of every rank.
#[derive(Clone, Debug)]
enum Reduction {
    LowLatency {
        inboxes: Vec<DeviceBufferSpanV1>,
        epochs: DeviceBufferSpanV1,
        elements: u64,
        capacity: u64,
        rank: u64,
        ranks: u64,
    },
    OneShot {
        inboxes: Vec<DeviceBufferSpanV1>,
        flags: Vec<DeviceBufferSpanV1>,
        epoch: DeviceBufferSpanV1,
        elements: u64,
        rank: u64,
        ranks: u64,
    },
    TwoShot(u64),
    /// A phased all-reduce entry, on its one state.
    Phased(u64),
}

impl Reduction {
    /// Each CTA's steps as `allreduce_oneshot` (`kernels-sm120`'s
    /// `comm.cuh`) takes them from the words' current values: wait for
    /// every peer's acknowledgment of the previous epoch, copy its slot and
    /// raise the data flag on each peer, wait for every source's data, then
    /// acknowledge each source and advance the tile's epoch.
    fn ctas(&self, words: &BTreeMap<Word, u32>, ranks_now: u64) -> Vec<Vec<CommOp>> {
        if let Reduction::LowLatency { .. } = self {
            return self.low_latency_ctas(words);
        }
        let Reduction::OneShot {
            inboxes,
            flags,
            epoch,
            elements,
            rank,
            ranks,
        } = self
        else {
            let (Reduction::TwoShot(index) | Reduction::Phased(index)) = self else {
                unreachable!()
            };
            let barrier = (u32::MAX, *index);
            return vec![vec![
                CommOp::Arrive(barrier),
                CommOp::Wait(barrier, ranks_now as u32),
            ]];
        };
        let (rank, ranks) = (*rank, *ranks);
        let tiles = elements.div_ceil(4096);
        for inbox in inboxes {
            assert!((rank + 1) * elements * 4 <= inbox.length, "{inbox:?}");
        }
        (0..tiles)
            .map(|tile| {
                let cell = word(*epoch, tile);
                let next = words.get(&cell).copied().unwrap_or(0) + 1;
                let index = rank * tiles + tile;
                let mut ops = Vec::new();
                for &dest in flags {
                    ops.push(CommOp::Wait(word(dest, ranks * tiles + index), next - 1));
                }
                for &dest in flags {
                    ops.push(CommOp::Release(word(dest, index), next));
                }
                let own = flags[rank as usize];
                for source in 0..ranks {
                    ops.push(CommOp::Wait(word(own, source * tiles + tile), next));
                }
                for source in 0..ranks {
                    let ack = ranks * tiles + source * tiles + tile;
                    ops.push(CommOp::Release(word(own, ack), next));
                }
                ops.push(CommOp::Release(cell, next));
                ops
            })
            .collect()
    }
}

impl Reduction {
    /// Each CTA's steps as `allreduce` (`kernels-sm120`'s
    /// `comm/allreduce_ll_bf16.cuh`) takes them, with thread 0's flag word
    /// standing for its slot (every slot's last word must lie in the
    /// inbox): at each recursive-doubling step, write every input it holds
    /// into the partner's slots of the epoch's parity, wait for the
    /// partner's and clear them; then advance the CTA's epoch.
    fn low_latency_ctas(&self, words: &BTreeMap<Word, u32>) -> Vec<Vec<CommOp>> {
        let Reduction::LowLatency {
            inboxes,
            epochs,
            elements,
            capacity,
            rank,
            ranks,
        } = self
        else {
            unreachable!()
        };
        let (rank, ranks, capacity) = (*rank, *ranks, *capacity);
        // Four 8-byte words per thread, 256 threads: 2048 u32 words per slot.
        let slot_words = 2048;
        let slot = |owner: u64, parity: u64, source: u64, cta: u64| {
            let first = ((parity * ranks + source) * capacity + cta) * slot_words;
            word(inboxes[owner as usize], first + slot_words - 1);
            // Thread 0's first value's flag half.
            word(inboxes[owner as usize], first + 1)
        };
        (0..elements / crate::sm120::engines::Engines::LL_SLICE_ELEMENTS)
            .map(|cta| {
                assert!(cta < capacity, "CTA {cta} past the capacity {capacity}");
                let cell = word(*epochs, cta);
                let next = words.get(&cell).copied().unwrap_or(0) + 1;
                let flag = 0x8000_0000 | next;
                let parity = u64::from(next & 1);
                let mut ops = Vec::new();
                let mut span = 1;
                while span < ranks {
                    let partner = rank ^ span;
                    let mine = rank & !(span - 1);
                    let theirs = partner & !(span - 1);
                    for i in 0..span {
                        ops.push(CommOp::Release(slot(partner, parity, mine + i, cta), flag));
                    }
                    for i in 0..span {
                        let own = slot(rank, parity, theirs + i, cta);
                        ops.push(CommOp::Wait(own, flag));
                        ops.push(CommOp::Release(own, 0));
                    }
                    span <<= 1;
                }
                ops.push(CommOp::Release(cell, next));
                ops
            })
            .collect()
    }
}

/// Every rank's all-reduce launches of the reduce stage in each phase of
/// `phases`, in stream order, through the lowering.
fn reductions(phases: &[Phase], low_latency: bool, phased: bool) -> Vec<Vec<(u32, Reduction)>> {
    let shape = Shape::TP4;
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| !has_mtp_layer(format))
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let ops = crate::ops::Ops::for_tests(shape, 1);
    let ranks = (0..shape.ranks())
        .map(|rank| {
            let mut buffers = rank_buffers(&device, rank, &format);
            buffers.comm.low_latency = low_latency;
            buffers.comm.phased = phased;
            buffers
        })
        .collect::<Vec<_>>();
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    let stage = crate::program::Stage::Reduce(0, crate::program::Sublayer::Attention);
    let mut out = vec![Vec::new(); ranks.len()];
    let mut collectives = vec![0; ranks.len()];
    for &phase in phases {
        for (rank, buffers) in ranks.iter().enumerate() {
            let inputs = inputs(rank as u32, phase);
            let lowering = Lowering {
                shape,
                format: &format,
                buffers,
                inputs: &inputs,
                ops: &ops,
                rank: rank as u32,
                index_topk: shape.profile.index_topk_max,
                pieces: &[],
                row_offset: 0,
                peers: &peers,
                head_rows: head_rows(phase),
                context_parallel: None,
                draft: None,
                verify_tokens: None,
                projections: None,
            };
            for launch in lowering.lower(phase, stage).unwrap().unwrap() {
                let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                let spans = |index: usize| match &arguments[index] {
                    KernelArgumentV1::DevicePointerArray(spans) => {
                        spans.iter().map(|span| span.unwrap()).collect::<Vec<_>>()
                    }
                    other => panic!("{other:?}"),
                };
                let span = |index: usize| match &arguments[index] {
                    KernelArgumentV1::DeviceBuffer(span) => *span,
                    other => panic!("{other:?}"),
                };
                let reduction = match launch.symbol.as_str() {
                    selected::LL_SYMBOL => {
                        let KernelArgumentV1::ScalarU64(elements) = arguments[4] else {
                            panic!("{:?}", arguments[4]);
                        };
                        let KernelArgumentV1::ScalarU32(capacity) = arguments[5] else {
                            panic!("{:?}", arguments[5]);
                        };
                        Reduction::LowLatency {
                            inboxes: spans(2),
                            epochs: span(3),
                            elements,
                            capacity: capacity.into(),
                            rank: rank as u64,
                            ranks: shape.ranks().into(),
                        }
                    }
                    selected::ONE_SHOT_SYMBOL => {
                        let KernelArgumentV1::ScalarU64(elements) = arguments[5] else {
                            panic!("{:?}", arguments[5]);
                        };
                        Reduction::OneShot {
                            inboxes: spans(2),
                            flags: spans(3),
                            epoch: span(4),
                            elements,
                            rank: rank as u64,
                            ranks: shape.ranks().into(),
                        }
                    }
                    selected::SCATTER_SYMBOL => {
                        // The two-shot's own state, apart from every
                        // one-shot state.
                        let two_shot = &buffers.comm.two_shot;
                        assert_eq!(span(4), two_shot.epoch);
                        assert_eq!(spans(3)[rank], two_shot.flags);
                        collectives[rank] += 1;
                        Reduction::TwoShot(collectives[rank])
                    }
                    symbol if selected::PHASED_SYMBOLS.contains(&symbol) => {
                        // The phased all-reduce's one state.
                        let state = &buffers.comm.phased_state;
                        assert_eq!(span(5), state.epoch);
                        assert_eq!(spans(4)[rank], state.flags);
                        assert_eq!(spans(2)[rank], state.inbox);
                        collectives[rank] += 1;
                        Reduction::Phased(collectives[rank])
                    }
                    _ => continue,
                };
                out[rank].push((phase.rows(), reduction));
            }
        }
    }
    out
}

/// Runs every rank's reductions in stream order with the words retained
/// across them, each rank's CTAs of a launch advancing as far as their
/// waits allow; panics when no rank can advance (a deadlock).
fn simulate(ranks: &[Vec<(u32, Reduction)>]) {
    let mut words = BTreeMap::<Word, u32>::new();
    let count = ranks.len() as u64;
    // Per rank: the launch it runs and its CTAs' remaining steps.
    let mut at = vec![0; ranks.len()];
    let mut running: Vec<Option<Vec<std::collections::VecDeque<CommOp>>>> = vec![None; ranks.len()];
    loop {
        let mut progress = false;
        for rank in 0..ranks.len() {
            if at[rank] == ranks[rank].len() {
                continue;
            }
            let ctas = running[rank].get_or_insert_with(|| {
                progress = true;
                ranks[rank][at[rank]]
                    .1
                    .ctas(&words, count)
                    .into_iter()
                    .map(Into::into)
                    .collect()
            });
            for cta in ctas.iter_mut() {
                while let Some(&op) = cta.front() {
                    match op {
                        CommOp::Wait(word, value) => {
                            if words.get(&word).copied().unwrap_or(0) != value {
                                break;
                            }
                        }
                        CommOp::Release(word, value) => {
                            words.insert(word, value);
                        }
                        CommOp::Arrive(word) => *words.entry(word).or_default() += 1,
                    }
                    cta.pop_front();
                    progress = true;
                }
            }
            if ctas.iter().all(|cta| cta.is_empty()) {
                running[rank] = None;
                at[rank] += 1;
                progress = true;
            }
        }
        if at
            .iter()
            .zip(ranks)
            .all(|(&at, launches)| at == launches.len())
        {
            return;
        }
        assert!(
            progress,
            "deadlock: each rank's launch {:?} of rows {:?}",
            at,
            at.iter()
                .zip(ranks)
                .map(|(&at, launches)| launches.get(at).map(|launch| launch.0))
                .collect::<Vec<_>>()
        );
    }
}

/// A sequence of steps that changes the row count and switches between
/// the low-latency entry (up to `BF16_LOW_LATENCY_ROWS` rows), the phased
/// entries (decode and verify buckets `engines::bf16_tuned` gives them)
/// and the pair, and, with either spinning entry refused at create, between
/// the others, every rank's comm words retained across it as a served
/// instance's are, runs to the end through the kernels' flag protocols
/// (the phased entries and the pair as collectives every rank reaches in
/// the same order): each one-shot row count takes its own one-shot state,
/// whose flag offsets and epoch cells fit its tiles, every low-latency row
/// count the one low-latency state and every phased call the one phased
/// state. A captured graph binds the same launches, so replaying a bucket
/// takes the same states.
#[test]
fn one_shot_states_survive_row_count_changes() {
    for (low_latency, phased) in [(true, true), (false, true), (true, false), (false, false)] {
        row_count_changes(low_latency, phased);
    }
}

fn row_count_changes(low_latency: bool, phased: bool) {
    let phases = [
        Phase::Decode(1),
        Phase::Decode(2),
        Phase::Prefill { rows: 2 },
        Phase::Decode(1),
        Phase::Verify { rows: 4, items: 1 },
        Phase::Decode(8),
        Phase::Prefill { rows: 16 },
        Phase::Verify { rows: 32, items: 8 },
        Phase::Verify { rows: 16, items: 4 },
        Phase::Prefill { rows: 2048 },
        Phase::Decode(2),
        Phase::Decode(4),
        Phase::Decode(1),
    ];
    let ranks = reductions(&phases, low_latency, phased);
    let rows = ranks[0].iter().map(|launch| launch.0).collect::<Vec<_>>();
    assert_eq!(rows, [1, 2, 2, 1, 4, 8, 16, 32, 16, 2048, 2, 4, 1]);
    // Each phase reduces through the entry the table gives it, the pair in
    // place of a refused one; the low-latency entry on one state.
    let mut epochs = BTreeMap::new();
    for rank in &ranks {
        for (&phase, (rows, reduction)) in phases.iter().zip(rank) {
            let want = match selected::bf16_tuned(*rows) {
                selected::Bf16Tuned::LowLatency if low_latency => "low-latency",
                _ if phased => "phased",
                _ => "pair",
            };
            let got = match reduction {
                Reduction::LowLatency { .. } => "low-latency",
                Reduction::Phased(_) => "phased",
                Reduction::OneShot { .. } => "one-shot",
                Reduction::TwoShot(_) => "pair",
            };
            assert_eq!(got, want, "{phase:?}");
            if let Reduction::LowLatency { epochs: span, .. } = reduction {
                epochs.insert(span.buffer.slot(), ());
            }
        }
    }
    assert_eq!(epochs.len(), if low_latency { ranks.len() } else { 0 });
    // Distinct row counts never share a one-shot flag or epoch buffer.
    let mut owners = BTreeMap::new();
    for (rows, reduction) in ranks.iter().flatten() {
        if let Reduction::OneShot { flags, epoch, .. } = reduction {
            for span in flags.iter().chain([epoch]) {
                assert_eq!(*owners.entry(span.buffer.slot()).or_insert(*rows), *rows);
            }
        }
    }
    simulate(&ranks);
}

/// One latent attention launch of a journal: its symbol, rows, CTA groups
/// per row and scalar arguments.
#[derive(Debug)]
struct LatentLaunch {
    symbol: String,
    rows: u32,
    groups: u32,
    scalars: Vec<u64>,
}

fn latent_launches(lines: &[String]) -> Vec<LatentLaunch> {
    let field = |line: &str, name: &str| -> u64 {
        let rest = &line[line.find(name).unwrap() + name.len()..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
        rest[..end].parse().unwrap()
    };
    lines
        .iter()
        .filter(|line| line.starts_with("launch r0 LatentAttention("))
        .map(|line| {
            let words = line.split(' ').collect::<Vec<_>>();
            let arguments = &line[line.find(" [").unwrap()..];
            LatentLaunch {
                symbol: words[3].to_owned(),
                rows: field(line, " rows ") as u32,
                groups: field(line, "grid_z: ") as u32,
                scalars: arguments
                    .split("Scalar")
                    .skip(1)
                    .map(|scalar| field(scalar, "("))
                    .collect(),
            }
        })
        .collect()
}

/// A row's latent attention output does not depend on the rows beside it:
/// at every decode and verify bucket the attention of each sparse layer and
/// its merge launch the same symbols with the same scalars (the paging, the
/// key budget, the 64 splits and the softmax scale; the merge's 64 splits),
/// so every row cuts its keys into the same splits and folds them in the
/// same order. Only the rows, the CTA groups per row (a power of two that
/// divides the splits, fewer as the bucket grows) and the bindings follow
/// the bucket.
#[test]
fn latent_rows_split_alike_at_every_bucket() {
    for kv in FORMATS {
        for phases in [
            [1, 2, 4, 8, 16, 32].map(Phase::Decode),
            [(1, 1), (4, 1), (8, 2), (16, 4), (32, 8), (32, 32)]
                .map(|(rows, items)| Phase::Verify { rows, items }),
        ] {
            let mut first: Option<Vec<LatentLaunch>> = None;
            let mut groups = BTreeSet::new();
            for phase in phases {
                let launches = latent_launches(&journal(kv, phase));
                // Eleven sparse layers, each an attention and its merge.
                assert_eq!(launches.len(), 22, "{kv:?} {phase:?}");
                for pair in launches.chunks(2) {
                    let (main, merge) = (&pair[0], &pair[1]);
                    // The split merge runs as its column merge: CTAs of
                    // 128 latent columns, four at latent 512.
                    assert_eq!(
                        merge.symbol,
                        main.symbol.replace("_v1", "_merge_columns_v1")
                    );
                    assert_eq!((main.rows, merge.rows), (phase.rows(), phase.rows()));
                    assert_eq!(
                        main.scalars[3],
                        u64::from(crate::sm120::engines::Engines::LATENT_MAX_SPLITS),
                        "{main:?}"
                    );
                    assert_eq!(
                        merge.scalars,
                        [u64::from(crate::sm120::engines::Engines::LATENT_MAX_SPLITS)]
                    );
                    assert_eq!(merge.groups, 4);
                    assert!(
                        crate::sm120::engines::Engines::LATENT_MAX_SPLITS
                            .is_multiple_of(main.groups),
                        "{main:?}"
                    );
                    groups.insert(main.groups);
                }
                match &first {
                    None => first = Some(launches),
                    Some(first) => {
                        for (a, b) in first.iter().zip(&launches) {
                            assert_eq!(
                                (&a.symbol, &a.scalars),
                                (&b.symbol, &b.scalars),
                                "{kv:?} {phase:?}"
                            );
                        }
                    }
                }
            }
            // The buckets do change the grid.
            assert!(groups.len() > 1, "{kv:?}: {groups:?}");
        }
    }
}

/// A drafter row's latent attention runs the 64 fixed splits in a draft
/// step of one row per item and in a catch-up pass on the verify path up
/// to 32 rows and on the prefill path above (five six-row items make 30
/// rows; a sixth item, 36) up to `DRAFT_SPLIT_ROWS` (64;
/// `Lowering::drafter_splits`). Each launches the decode or the verify
/// entry, whose kernels are the same, with the same scalars (the paging,
/// the key budget, the 64 splits and the softmax scale) and its merge, so
/// a row's bits follow only its own query and keys there. A prefill-path pass of
/// more than `DRAFT_SPLIT_ROWS` rows (96 here, or a prefill chunk's run over
/// every completed row) keeps prefill's one split per row and no merge. On
/// FP8 pages that launch is the entry's slice-widened form, which writes the
/// entry's bits from the same ABI and grid (kernel-lab
/// `latent_attention_slices`); it compares as the entry.
#[test]
fn drafter_rows_take_the_fixed_splits_in_narrow_passes() {
    for kv in FORMATS {
        let mut first: Option<LatentLaunch> = None;
        for (pass, phase, kernel) in [
            (
                DraftPass::Extend,
                Phase::Verify { rows: 30, items: 8 },
                "verify",
            ),
            (
                DraftPass::Extend,
                Phase::Verify { rows: 32, items: 8 },
                "verify",
            ),
            (DraftPass::Extend, Phase::Prefill { rows: 36 }, "verify"),
            (DraftPass::Extend, Phase::Prefill { rows: 64 }, "verify"),
            (DraftPass::Extend, Phase::Prefill { rows: 96 }, "prefill"),
            (DraftPass::Extend, Phase::Prefill { rows: 2048 }, "prefill"),
            (DraftPass::Step, Phase::Decode(1), "decode"),
            (DraftPass::Step, Phase::Decode(8), "decode"),
        ] {
            let format = Shape::TP4
                .kv_formats()
                .into_iter()
                .find(|format| latent_format(format) == kv && has_mtp_layer(format));
            if format.is_none() {
                continue;
            }
            let launches = latent_launches(&journal_of(kv, phase, Some(pass)));
            let label = format!("{kv:?} {pass:?} {phase:?}: {launches:?}");
            let launch = &launches[0];
            let symbol = launch.symbol.replace("_slices_v1", "_v1");
            assert!(symbol.ends_with(&format!("_{kernel}_v1")), "{label}");
            assert_eq!(launch.rows, phase.rows());
            if kernel == "prefill" {
                let slices = launch.symbol.ends_with("_prefill_slices_v1");
                assert_eq!(slices, kv == LatentKvFormat::Fp8E4M3Block128, "{label}");
                assert_eq!(launches.len(), 1, "{label}");
                assert_eq!((launch.groups, launch.scalars[3]), (1, 1), "{label}");
                continue;
            }
            assert_eq!(launches.len(), 2, "{label}");
            let merge = &launches[1];
            assert_eq!(
                merge.symbol,
                launch.symbol.replace("_v1", "_merge_columns_v1")
            );
            assert_eq!(
                launch.scalars[3],
                u64::from(crate::sm120::engines::Engines::LATENT_MAX_SPLITS),
                "{label}"
            );
            assert_eq!(
                merge.scalars,
                [u64::from(crate::sm120::engines::Engines::LATENT_MAX_SPLITS)],
                "{label}"
            );
            match &first {
                None => first = Some(launches.into_iter().next().unwrap()),
                Some(first) => assert_eq!(first.scalars, launch.scalars, "{label}"),
            }
        }
    }
}
