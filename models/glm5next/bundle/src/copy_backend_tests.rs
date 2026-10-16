//! The stream expansion and the router's score compaction launch the
//! shared row copies (`model_glm5next_contract::copies`) where GLM's own
//! rows ran, and the launches change only there: each grids and records
//! the smallest row bucket holding the phase's rows, as the own rows did,
//! with PDL, binding the same source, destination and live count spans,
//! under the shared symbol and with the shared entry's shape arguments
//! after the spans. These are the launches the re-pinned journals change.

use bundle_runtime::dispatch;
use bundle_runtime::fake_host::FakeDevice;
use bundle_runtime::geometry::Items;
use bundle_runtime::graphs::verify_tokens;
use bundle_runtime::kv::KvFormat;
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::upload::DeviceInputs;
use kernel_core::contract::LaunchPlan;
use kernel_core::latent_attention::LatentKvFormat;
use model_glm5next_contract::contract::{
    DECODE_BATCH, Glm, MAX_VERIFY_ROWS, PREFILL_TOKENS, Shape, has_mtp_layer, latent_format,
};
use model_glm5next_contract::sm120::engines::{BROADCAST_SYMBOL, SLICE_SYMBOL};
use runtime_bundle_abi::{DeviceBufferSpanV1, KernelArgumentV1};

use crate::buffers::{Buffer, CommState, RankBuffers};
use crate::journal_tests::{head_rows, inputs_within, rank_buffers_serving};
use crate::lower::Lowering;
use crate::ops::Ops;
use crate::program::{Phase, Stage};

/// Every phase an instance serving `DECODE_BATCH` items with
/// `MAX_VERIFY_ROWS` verify rows per item runs a step of
/// (`Instance::phase_of`): an eager prefill of each row count up to
/// `PREFILL_TOKENS`, a decode graph of each item bucket, and a verify step
/// of each item bucket with each token count up to its verify rows and its
/// verify tokens.
fn served_phases(items: Items) -> Vec<Phase> {
    let mut phases: Vec<Phase> = (1..=PREFILL_TOKENS)
        .map(|rows| Phase::Prefill { rows })
        .collect();
    phases.extend(items.buckets().map(Phase::Decode));
    let tokens = verify_tokens::<Glm>(items);
    for bucket in items.buckets() {
        let most = (bucket * MAX_VERIFY_ROWS).min(tokens);
        phases.extend((1..=most).map(|rows| Phase::Verify {
            rows,
            items: bucket,
        }));
    }
    phases
}

/// One phase of each kind, whose inputs every phase of the kind reads.
const KINDS: [Phase; 3] = [
    Phase::Prefill { rows: 2048 },
    Phase::Decode(8),
    Phase::Verify { rows: 32, items: 8 },
];

/// Every rank's buffers of `format` in an instance serving `items` with
/// `MAX_VERIFY_ROWS` verify rows per item, the ops it prepares (group size
/// 1) and each rank's inputs of each phase kind.
struct Rig {
    shape: Shape,
    format: KvFormat,
    ops: Ops,
    ranks: Vec<RankBuffers>,
    peers: Vec<CommState>,
    inputs: Vec<[DeviceInputs; 3]>,
}

impl Rig {
    fn new(format: KvFormat, items: Items) -> Rig {
        let shape = Shape::TP4;
        let device = FakeDevice::default();
        let ranks = (0..shape.ranks())
            .map(|rank| rank_buffers_serving(&device, rank, &format, items, MAX_VERIFY_ROWS))
            .collect::<Vec<_>>();
        let peers = ranks.iter().map(|buffers| buffers.comm.clone()).collect();
        Rig {
            shape,
            format,
            ops: Ops::for_tests(shape, 1),
            ranks,
            peers,
            inputs: (0..shape.ranks())
                .map(|rank| KINDS.map(|kind| inputs_within(rank, kind, items)))
                .collect(),
        }
    }

    /// Rank `rank`'s lowering of the target's `phase`.
    fn lowering(&self, rank: u32, phase: Phase) -> Lowering<'_> {
        let kind = match phase {
            Phase::Prefill { .. } => 0,
            Phase::Decode(_) => 1,
            Phase::Verify { .. } => 2,
        };
        Lowering {
            shape: self.shape,
            format: &self.format,
            buffers: &self.ranks[rank as usize],
            inputs: &self.inputs[rank as usize][kind],
            ops: &self.ops,
            rank,
            index_topk: self.shape.profile.index_topk_max,
            pieces: &[],
            row_offset: 0,
            peers: &self.peers,
            head_rows: head_rows(phase),
            context_parallel: None,
            draft: None,
            verify_tokens: None,
            projections: None,
        }
    }
}

/// The plan of a copy over `bucket` rows with `threads` threads per CTA,
/// the own rows' plan.
fn plan(bucket: u32, threads: u32) -> LaunchPlan {
    LaunchPlan {
        grid: [bucket, 1, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    }
}

/// `launch` is `symbol` over the phase's bucket, reading `source`, writing
/// `destination` and reading the live count, then `scalars`.
fn check(
    lowering: &Lowering<'_>,
    phase: Phase,
    launch: &PreparedLaunch,
    (symbol, threads): (&str, u32),
    (source, destination): (Buffer, Buffer),
    scalars: &[u32],
) {
    // The row bucket the phase's norms grid, as the own rows did.
    let bucket = lowering
        .norm(phase, Stage::FinalNorm)
        .unwrap()
        .launch
        .plan
        .grid[0];
    let what = format!("r{} {phase:?} {symbol}", lowering.rank);
    assert_eq!(launch.symbol, symbol, "{what}");
    assert_eq!(launch.rows, bucket, "{what}");
    assert_eq!(launch.plan, plan(bucket, threads), "{what}");
    let span = |span: DeviceBufferSpanV1| KernelArgumentV1::DeviceBuffer(span);
    let mut expected = vec![
        span(lowering.scratch(source)),
        span(lowering.scratch(destination)),
        span(lowering.inputs.live_tokens),
    ];
    expected.extend(
        scalars
            .iter()
            .map(|&value| KernelArgumentV1::ScalarU32(value)),
    );
    assert_eq!(
        dispatch::arguments(&launch.arguments, &launch.bindings).unwrap(),
        expected,
        "{what}"
    );
}

/// At every served phase on every rank, the expansion broadcasts the
/// reduced embedding rows into the first sublayer's streams (512 16-byte
/// vectors per 4096-wide BF16 row, four streams), and the compaction
/// slices the router's 288 expert scores from its 320 padded F32 columns
/// (576 of 640 16-bit words per row, from word 0).
#[test]
fn the_copies_launch_the_shared_entries_over_the_own_rows_buffers() {
    let format = Shape::TP4
        .kv_formats()
        .into_iter()
        .find(|format| {
            latent_format(format) == LatentKvFormat::Fp8E4M3Block128 && has_mtp_layer(format)
        })
        .unwrap();
    let items = Items::new(DECODE_BATCH).unwrap();
    let rig = Rig::new(format, items);
    let p = rig.shape.profile;
    assert_eq!((p.hidden, p.hc_streams), (4096, 4));
    let mut checked = 0;
    for phase in served_phases(items) {
        for rank in 0..rig.shape.ranks() {
            let lowering = rig.lowering(rank, phase);
            let [expand] = &lowering
                .lower(phase, Stage::ExpandStreams)
                .unwrap()
                .unwrap()[..]
            else {
                panic!("r{rank} {phase:?}: one expansion launch");
            };
            check(
                &lowering,
                phase,
                expand,
                (BROADCAST_SYMBOL, 256),
                (Buffer::Reduced, Buffer::Streams0),
                &[512, 4],
            );
            let compact = lowering.compact_scores(phase).unwrap().launch;
            check(
                &lowering,
                phase,
                &compact,
                (SLICE_SYMBOL, 256),
                (Buffer::RouterScores, Buffer::RouterLogits),
                &[576, 640, 1, 0, 576, 1, 0],
            );
            checked += 1;
        }
    }
    assert!(checked > 4 * 8192, "{checked}");
}
