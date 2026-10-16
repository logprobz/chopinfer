//! The context-parallel lowering at group sizes 2 and 4 on the fake device
//! (`lower/context_parallel.rs`): pinned launch journals per size and phase
//! in the form of `journal_tests.rs` (whose size-1 journals stay pinned
//! unchanged), the group launches' bindings, and create at the served sizes.

use std::collections::{BTreeMap, BTreeSet};

use bundle_runtime::dispatch;
use bundle_runtime::fake_host::{FakeDevice, check_journal};
use kernel_core::latent_attention::LatentKvFormat;
use model_glm5next_contract::contract::{Shape, has_mtp_layer, latent_format};
use model_glm5next_contract::sm120::context_parallel::{ALLGATHER_SYMBOL, PHASED_SYMBOL};
use runtime_bundle_abi::{
    BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesTableV1, KernelArgumentV1,
};

use crate::buffers::RankBuffers;
use crate::context_parallel::{ContextParallel, RankState, bucket_index};
use crate::journal_tests::{
    FORMATS, PHASES, check_operands, extents, head_rows, inputs, planner, rank_buffers,
};
use crate::launch::records;
use crate::lower::Lowering;
use crate::program::{Phase, Stage};

/// Every rank's context-parallel state for groups of `size`, allocated on
/// the fake device after every rank's buffers, as create does after peer
/// access.
pub(crate) fn group_state(device: &FakeDevice, size: u32) -> ContextParallel {
    let dims = model_glm5next_contract::context_parallel::Dims::of(Shape::TP4);
    let bytes = model_glm5next_contract::context_parallel::rank_bytes(dims, size).unwrap();
    let ranks = (0..Shape::TP4.ranks())
        .map(|rank| {
            let stream = BundleStreamHandleV1::from_parts(1, 3000 + rank);
            RankState::allocate(&bytes, |bytes| {
                let id = device.id();
                Ok::<_, ()>(DeviceBufferSpanV1::new(
                    device.allocate(id, stream, bytes, 256),
                    0,
                    bytes,
                ))
            })
            .unwrap()
        })
        .collect();
    ContextParallel { size, ranks }
}

/// One rank's lowered launch: its stage, symbol, rows and bound
/// arguments.
struct Launch {
    stage: Stage,
    symbol: String,
    rows: u32,
    arguments: Vec<KernelArgumentV1>,
}

/// Every rank's launches of `phase` over `kv` in groups of `size`, with
/// the journal lines of the fake device: the program's records, the
/// lowered stages as the launches the device would receive.
fn lowered(kv: LatentKvFormat, phase: Phase, size: u32) -> (Vec<String>, Vec<Vec<Launch>>) {
    let shape = Shape::TP4;
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| latent_format(format) == kv && !has_mtp_layer(format))
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let records = records(shape, kv, phase).unwrap();
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<RankBuffers>>();
    let cp = group_state(&device, size);
    let ops = crate::ops::Ops::for_tests(shape, size);
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    for (rank, buffers) in ranks.iter().enumerate() {
        let rank = rank as u32;
        let inputs = inputs(rank, phase);
        let mut extents = extents(rank, buffers, &peers, phase);
        for span in cp.ranks.iter().flat_map(RankState::spans) {
            let end = extents.entry(span.buffer.slot()).or_default();
            *end = (*end).max(span.offset + span.length);
        }
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
            context_parallel: Some(&cp),
            draft: None,
            verify_tokens: None,
            projections: None,
        };
        let mut launches = Vec::new();
        for record in &records {
            let Some(lowered) = lowering.lower(phase, record.stage).unwrap() else {
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
            for launch in lowered {
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
                launches.push(Launch {
                    stage: record.stage,
                    symbol: launch.symbol,
                    rows: launch.rows,
                    arguments,
                });
            }
        }
        out.push(launches);
    }
    (device.journal.take(), out)
}

/// The size-1 journals stay pinned in `journal_tests.rs`; at sizes 2 and 4
/// every rank allocates its group state after its buffers, the latent KV
/// write localizes its positions first, the indexer selects per rank and
/// merges the group's pairs, and the attention runs the group's stages.
/// Re-pinned when each pair-gather bucket took its own output (eleven more
/// allocations per rank, and the merge's pairs binding).
/// Re-pinned when the partials moved by one all-gather per owner and a group
/// barrier ended each piece (each sparse layer's exchange became `size`
/// owner gathers and the barrier; each rank allocates the barrier's word,
/// output and state and a partial gather state per bucket instead of the
/// exchange's, and its receive buffer holds `size` blocks).
/// Re-pinned when the MoE took the F32 routed sum (its symbol, the F32 MLP
/// output and the F32 outputs of the experts' and shared expert's down
/// records).
/// Re-pinned when the MTP drafter's stream and input buffers joined the
/// scratch (one more allocation per rank).
/// Re-pinned when the drafter's scratch moved to the formats that hold its
/// layer (four fewer allocations per rank here).
/// Re-pinned when the row-split projections' records wrote F32 partials
/// (their output dtype), and when the lowered BF16 projections' weight
/// views took their zero rows to the row tile (the views' offsets), and
/// when the KDA gates' low-rank buffers joined the scratch (two more
/// allocations per rank), and when the router's padded scores took their
/// own buffer (one more), and when the latent attention kept every row's
/// 64 fixed splits (the decode and verify attention's splits and outputs,
/// and their merges' splits), and when every mixer ran the mix-collapse
/// split over CTAs (its symbol, grid, chain sums and tickets: two more
/// allocations per rank).
/// Re-pinned when the pair and query all-gathers took the phased group
/// gather and each piece's `size` owner gathers became one phased exchange
/// (every other launch, argument and allocation unchanged:
/// `$CHOPINFER_DATA/glm5next/PF5/r1/journals.py`).
/// Re-pinned when prefill mixers of 1024 rows and more ran the one-CTA
/// mix-collapse (the prefill mixer launches' symbol, grid and bindings).
/// Re-pinned when the KDA prefill ran its tiled prepare and scan (their
/// symbols and dynamic shared bytes). Re-pinned when the phased
/// all-reduce's state joined the comm state (three more allocations per
/// rank, the handles after them).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse and
/// its two symbols joined create (the handles).
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies (identity change, accuracy unaffected: their symbols and
/// shape arguments).
#[test]
fn group_journals_are_pinned_per_size_and_phase() {
    let kv = FORMATS[0];
    let pinned: [((u32, Phase), (usize, u64)); 6] = [
        ((2, PHASES[0]), (6564, 0x8076fee80e66881b)),
        ((2, PHASES[1]), (5860, 0x4d3a4633c8145a10)),
        ((2, PHASES[2]), (5860, 0x8b0d03b9a30ddc7e)),
        ((4, PHASES[0]), (6652, 0x2fd2fb3d38ffc5bd)),
        ((4, PHASES[1]), (6036, 0x883f61d637e077ec)),
        ((4, PHASES[2]), (6036, 0xdfc3f3b73cc94ab8)),
    ];
    let mut failures = Vec::new();
    for ((size, phase), expected) in pinned {
        let (lines, _) = lowered(kv, phase, size);
        let file = format!("glm5next-journal-cp{size}-{kv:?}-{}.txt", phase.rows());
        if let Err(error) = check_journal(&lines, expected, &file) {
            failures.push(format!("size {size} {phase:?}: {error}"));
        }
    }
    // The BF16 pages lower alike.
    for phase in PHASES {
        lowered(FORMATS[1], phase, 2);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The handle slots of rank `rank`'s state.
fn slots(cp: &ContextParallel, rank: u32) -> BTreeSet<u32> {
    cp.ranks[rank as usize]
        .spans()
        .iter()
        .map(|span| span.buffer.slot())
        .collect()
}

fn pointers(argument: &KernelArgumentV1) -> Vec<DeviceBufferSpanV1> {
    match argument {
        KernelArgumentV1::DevicePointerArray(spans) => spans.iter().flatten().copied().collect(),
        other => panic!("{other:?}"),
    }
}

/// Each group launch addresses only its group's members, in group-rank
/// order; each rank writes only its own pages; each sparse layer runs one
/// partial attention per source block; and each bucket keeps its own
/// gather flags.
#[test]
fn group_launches_address_their_members_and_own_pages() {
    let kv = FORMATS[0];
    for size in [2_u32, 4] {
        let device: FakeDevice = FakeDevice::default();
        // The slots `lowered` allocates: the same order on a fresh device.
        let cp = {
            let format = Shape::TP4
                .kv_formats()
                .into_iter()
                .find(|format| latent_format(format) == kv && !has_mtp_layer(format))
                .unwrap();
            for rank in 0..Shape::TP4.ranks() {
                rank_buffers(&device, rank, &format);
            }
            group_state(&device, size)
        };
        let mut owners = BTreeMap::new();
        for phase in [
            Phase::Decode(8),
            Phase::Decode(1),
            Phase::Verify { rows: 32, items: 8 },
            Phase::Prefill { rows: 2048 },
        ] {
            let (_, ranks) = lowered(kv, phase, size);
            let sparse = Shape::TP4.sparse_layers() as usize;
            for (rank, launches) in ranks.iter().enumerate() {
                let rank = rank as u32;
                let first = rank / size * size;
                let members: Vec<u32> = (first..first + size).collect();
                let partials = launches
                    .iter()
                    .filter(|launch| {
                        // The prefill entry or its slice-widened form.
                        (launch.symbol.contains("_partial_v1")
                            || launch.symbol.contains("_partial_slices_v1"))
                            && matches!(launch.stage, Stage::LatentAttention(_))
                    })
                    .count();
                assert_eq!(partials, sparse * size as usize, "rank {rank}");
                for launch in launches {
                    if let Stage::LatentKvWrite(_) = launch.stage
                        && launch.symbol.contains("kv_write")
                    {
                        // The pool is the rank's own; the positions are
                        // its localized ones.
                        let own = slots(&cp, rank);
                        assert!(launch.arguments.iter().any(|argument| matches!(
                            argument,
                            KernelArgumentV1::DeviceBuffer(span) if own.contains(&span.buffer.slot())
                        )));
                    }
                    let (outputs, flags, epoch, words) = match launch.symbol.as_str() {
                        ALLGATHER_SYMBOL => (1, 2, 3, 4),
                        PHASED_SYMBOL => (2, 3, 4, 5),
                        _ => continue,
                    };
                    let KernelArgumentV1::ScalarU64(elements) = launch.arguments[words] else {
                        panic!("{:?}", launch.arguments[words]);
                    };
                    for index in [outputs, flags] {
                        let spans = pointers(&launch.arguments[index]);
                        assert_eq!(spans.len(), size as usize, "{}", launch.symbol);
                        for (span, member) in spans.iter().zip(&members) {
                            assert!(
                                slots(&cp, *member).contains(&span.buffer.slot()),
                                "rank {rank} {} slot {} of member {member}",
                                launch.symbol,
                                span.buffer.slot()
                            );
                        }
                    }
                    let KernelArgumentV1::DeviceBuffer(own_epoch) = launch.arguments[epoch] else {
                        panic!("{:?}", launch.arguments[epoch]);
                    };
                    assert!(slots(&cp, rank).contains(&own_epoch.buffer.slot()));
                    if elements > 1 {
                        // A gather state serves one bucket only; the
                        // barrier's one word serves every bucket.
                        let bucket = bucket_index(launch.rows).unwrap();
                        for span in pointers(&launch.arguments[flags])
                            .iter()
                            .chain([&own_epoch])
                        {
                            assert_eq!(
                                *owners.entry(span.buffer.slot()).or_insert(bucket),
                                bucket,
                                "size {size} {}",
                                launch.symbol
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The bytes of every device allocation the fake host journaled.
fn allocated(host: &crate::fake_host_tests::FakeHost) -> u64 {
    host.device
        .journal
        .borrow()
        .iter()
        .filter_map(|line| line.strip_prefix("allocate "))
        .map(|rest| rest.split(' ').next().unwrap().parse::<u64>().unwrap())
        .sum()
}

/// Create serves context-parallel sizes 2 and 4: it allocates every rank's
/// group state, exactly the bytes the capacity plan counts (`RankBytes`,
/// `capacity::workspace`) beyond a size-1 instance's, loads the
/// context-parallel symbols, and destroy gives everything back.
#[test]
fn create_serves_context_parallel_groups() {
    let record = |size: u32| {
        let mut tail = runtime_bundle_sdk::context_parallel::TAG.to_vec();
        tail.extend_from_slice(&size.to_le_bytes());
        tail.extend(crate::config::tests::options(2048));
        crate::config::tests::record(model_glm5next_contract::contract::DECODE_BATCH, &tail)
    };
    // A size-1 instance's allocations, with every prefill whole as a
    // group keeps it.
    let baseline = {
        let host = crate::fake_host_tests::FakeHost::new((3, 32));
        let mut instance = {
            crate::fake_host_tests::api().create(
                HostServicesTableV1::new(std::rc::Rc::clone(&host)),
                &crate::config::tests::record(
                    model_glm5next_contract::contract::DECODE_BATCH,
                    &crate::config::tests::options(2048),
                ),
            )
        }
        .unwrap();
        let bytes = allocated(&host);
        instance.destroy().unwrap();
        bytes
    };
    for size in [2, 4] {
        let host = crate::fake_host_tests::FakeHost::new((3, 32));
        let mut instance = crate::fake_host_tests::api()
            .create(
                HostServicesTableV1::new(std::rc::Rc::clone(&host)),
                &record(size),
            )
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        // Every allocation beyond a size-1 instance's is a rank's group
        // state, as the capacity plan counts it.
        let bytes = model_glm5next_contract::context_parallel::rank_bytes(
            model_glm5next_contract::context_parallel::Dims::of(Shape::TP4),
            size,
        )
        .unwrap();
        assert_eq!(
            allocated(&host) - baseline,
            4 * bytes.total(),
            "size {size}"
        );
        assert_eq!(
            bytes.total(),
            model_glm5next_contract::capacity::workspace(Shape::TP4.geometry, 4, size)
                .context_parallel
        );
        // Each rank's send and receive buffers: `size` blocks of 2048 rows.
        let block = u64::from(size) * crate::context_parallel::block_floats(Shape::TP4, 2048) * 4;
        assert_eq!((bytes.send, bytes.receive), (block, block));
        let line = format!("allocate {block} 256");
        let journal = host.device.journal.borrow().clone();
        assert_eq!(
            journal.iter().filter(|entry| **entry == line).count(),
            8,
            "size {size}"
        );
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// An eager prefill of 8,192 rows runs the group's selection and attention
/// per indexer piece. A piece's gather takes its bucket of rows from the
/// piece's first row, moved back to end inside the scratch: the last piece
/// (1,192 rows from row 7,000) gathers rows 6,144 to 8,192, its partial
/// attention reads each block 856 rows in, and its merge writes its own
/// rows.
#[test]
fn prefill_pieces_gather_their_bucket_inside_the_scratch() {
    let shape = Shape::TP4;
    let kv = FORMATS[0];
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| latent_format(format) == kv && !has_mtp_layer(format))
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<RankBuffers>>();
    let cp = group_state(&device, 2);
    let ops = crate::ops::Ops::for_tests(shape, 2);
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    let phase = Phase::Prefill { rows: 8192 };
    let rank = 1;
    let inputs = inputs(rank, phase);
    let pieces = [
        (0, 1500),
        (1500, 2048),
        (3548, 2048),
        (5596, 1404),
        (7000, 1192),
    ]
    .map(|(first, rows)| crate::lower::Piece {
        first,
        rows,
        inputs,
    });
    let mut extents = extents(rank, &ranks[rank as usize], &peers, phase);
    for span in cp.ranks.iter().flat_map(RankState::spans) {
        let end = extents.entry(span.buffer.slot()).or_default();
        *end = (*end).max(span.offset + span.length);
    }
    let lowering = Lowering {
        shape,
        format: &format,
        buffers: &ranks[rank as usize],
        inputs: &inputs,
        ops: &ops,
        rank,
        index_topk: shape.profile.index_topk_max,
        pieces: &pieces,
        row_offset: 0,
        peers: &peers,
        head_rows: head_rows(phase),
        context_parallel: Some(&cp),
        draft: None,
        verify_tokens: None,
        projections: None,
    };
    let row_bytes = 16 * 512 * 2;
    let own = &cp.ranks[rank as usize];
    for stage in [Stage::IndexSelect(3), Stage::LatentAttention(3)] {
        let launches = lowering.lower(phase, stage).unwrap().unwrap();
        let mut gathers = Vec::new();
        let mut queries = Vec::new();
        let mut merged = Vec::new();
        let mut exchanges = Vec::new();
        let mut barriers = 0;
        // Per piece, the launch kinds in order: (p)artial attention,
        // e(x)change, (m)erge, (b)arrier.
        let mut order = String::new();
        for launch in &launches {
            let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
            check_operands(rank, &launch.symbol, &arguments, &extents);
            let span = |index: usize| match arguments[index] {
                KernelArgumentV1::DeviceBuffer(span) => span,
                ref other => panic!("{other:?}"),
            };
            if launch.symbol == PHASED_SYMBOL && span(0).buffer == own.send.buffer {
                let KernelArgumentV1::ScalarU64(stride) = arguments[1] else {
                    panic!("{:?}", arguments[1]);
                };
                exchanges.push((stride, span(0).offset, span(0).length));
                order.push('x');
            } else if launch.symbol == ALLGATHER_SYMBOL
                && span(0).buffer == own.barrier_input.buffer
            {
                barriers += 1;
                order.push('b');
            } else if launch.symbol == PHASED_SYMBOL {
                assert_eq!(arguments[1], KernelArgumentV1::ScalarU64(0));
                gathers.push((launch.rows, span(0).offset, span(0).length));
            } else if launch.symbol.ends_with("_prefill_partial_v1")
                || launch.symbol.ends_with("_prefill_partial_slices_v1")
            {
                assert_eq!(span(0).buffer, own.queries.buffer);
                queries.push(span(0).offset);
                order.push('p');
            } else if launch.symbol.contains("partial_merge") {
                // The compact receive buffer, from its base.
                assert_eq!(
                    (span(0).buffer, span(0).offset, span(0).length),
                    (
                        own.receive.buffer,
                        own.receive.offset,
                        2 * 2048 * 16 * 514 * 4
                    )
                );
                merged.push(span(2).offset);
                order.push('m');
            }
        }
        if stage == Stage::IndexSelect(3) {
            // Each piece gathers its pairs from the pair buffer's start.
            assert_eq!(gathers.len(), 5);
            assert!(gathers.iter().all(|gather| gather.1 == 0));
            continue;
        }
        let query = ranks[rank as usize]
            .scratch
            .get(&crate::buffers::Buffer::QueryAbsorbed)
            .unwrap()
            .offset;
        assert_eq!(
            gathers,
            [
                (1500, query, 2048 * row_bytes),
                (2048, query + 1500 * row_bytes, 2048 * row_bytes),
                (2048, query + 3548 * row_bytes, 2048 * row_bytes),
                (1404, query + 5596 * row_bytes, 2048 * row_bytes),
                (1192, query + 6144 * row_bytes, 2048 * row_bytes),
            ]
        );
        // Every piece: both partial attentions, then one exchange of both
        // send blocks (the 2,048-row bucket's block words apart), the merge
        // and the barrier.
        assert_eq!(order, "ppxmb".repeat(5));
        assert_eq!(barriers, 5);
        let block = 2048 * 16 * 514;
        assert_eq!(exchanges, [(block, 0, 2 * block * 4)].repeat(5));
        // Two source blocks per piece; the last piece's rows start 856
        // rows into each block of its 2,048-row bucket.
        assert_eq!(queries.len(), 10);
        assert_eq!(&queries[8..], [856 * row_bytes, (2048 + 856) * row_bytes]);
        let out = ranks[rank as usize]
            .scratch
            .get(&crate::buffers::Buffer::AttentionOut)
            .unwrap()
            .offset;
        assert_eq!(
            merged,
            [0, 1500, 3548, 5596, 7000].map(|row: u64| out + row * row_bytes)
        );
    }
}

/// One decode step of request 7 in slot 0 after `previous` tokens, over the
/// page table `pages`.
fn decode_step(previous: u64, pages: Vec<u64>) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, true, 1, true, vec![11])],
        vec![BatchAppendV1::new(previous, 1, previous + 1, pages, None)],
    )
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

/// Served at size 2, a decode step takes the host's striped page table
/// (one local page per 128-token stripe) and refuses one of 64-token
/// pages; each rank then issues the group's launches.
#[test]
fn a_served_group_decodes_over_its_striped_pages() {
    let mut tail = runtime_bundle_sdk::context_parallel::TAG.to_vec();
    tail.extend_from_slice(&2_u32.to_le_bytes());
    tail.extend(crate::config::tests::options(2048));
    let record =
        crate::config::tests::record(model_glm5next_contract::contract::DECODE_BATCH, &tail);
    let host = crate::fake_host_tests::FakeHost::new((3, 32));
    let mut instance = crate::fake_host_tests::api()
        .create(HostServicesTableV1::new(std::rc::Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 200);
    crate::instance::run_lowered_stages_alone(handle);
    // 201 tokens: four 64-token pages, two 128-token stripes.
    assert_eq!(
        instance
            .execute(
                &decode_step(200, vec![0, 1, 2, 3]),
                &mut crate::fake_host_tests::NoRows::default()
            )
            .unwrap_err(),
        runtime_bundle_abi::BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    );
    let start = host.device.journal.borrow().len();
    instance
        .execute(
            &decode_step(200, vec![0, 1]),
            &mut crate::fake_host_tests::NoRows::default(),
        )
        .unwrap();
    let lines = crate::fake_host_tests::issued(&host, start);
    for symbol in [
        "localize_positions2",
        "select_per_rank2",
        "topk_merge2",
        "localize_keys2",
        "decode_partial_v1",
        "chop_comm_allgather_f32_v1",
        "partial_merge2",
    ] {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("launch ") && line.contains(symbol)),
            "{symbol}"
        );
    }
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A word of a fake buffer: its handle slot and byte offset.
type Word = (u32, u64);

/// One step of the simulated group launches.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Spin until the word holds the value.
    Wait(Word, u32),
    Release(Word, u32),
    /// Store gather `generation`'s tile from topology rank `writer` at the
    /// word (its first byte).
    Store(Word, u32, u32),
    /// Add one to a barrier's count, then wait for `members` arrivals.
    Arrive(Word),
    Barrier(Word, u32),
}

/// One rank's launch as the simulation runs it.
#[derive(Clone, Debug)]
enum Sim {
    /// The rank's `generation`th all-gather: its outputs and flags in slot
    /// order, its epoch, words per rank, group rank, target (the group size
    /// for every member) and topology rank.
    Gather {
        outputs: Vec<DeviceBufferSpanV1>,
        flags: Vec<DeviceBufferSpanV1>,
        epoch: DeviceBufferSpanV1,
        elements: u64,
        rank: u64,
        target: u64,
        writer: u32,
        generation: u32,
    },
    /// The rank's `generation`th phased group gather
    /// (`cuda/context_parallel/phased_gather.cuh`): its outputs and flags in
    /// slot order, its epoch, words per member, group rank and topology
    /// rank. `acknowledged` false drops its waits for the receivers'
    /// acknowledgements (a mutation the simulation must catch).
    Phased {
        outputs: Vec<DeviceBufferSpanV1>,
        flags: Vec<DeviceBufferSpanV1>,
        epoch: DeviceBufferSpanV1,
        elements: u64,
        rank: u64,
        writer: u32,
        generation: u32,
        acknowledged: bool,
    },
    /// A launch reading the output of the rank's last gather that delivered
    /// there: every source block's tiles must hold that gather's generation,
    /// written by the group member of that slot (topology rank `first +
    /// source`).
    Consume {
        output: DeviceBufferSpanV1,
        elements: u64,
        ranks: u64,
        first: u32,
        generation: u32,
        symbol: String,
    },
    /// The sublayer all-reduce between steps: a barrier of every rank.
    Reduce { key: Word, members: u32 },
}

const TILE_WORDS: u64 = 4096;

fn tile_word(span: DeviceBufferSpanV1, word: u64) -> Word {
    (span.buffer.slot(), span.offset + word * 4)
}

impl Sim {
    /// Each CTA's steps as `allgather` (`cuda/comm/comm.cuh`) takes them
    /// from the words' current values: acknowledge the previous call to
    /// every source, wait for each delivered destination's acknowledgment
    /// (every member, or the target only), store the tile and raise its flag
    /// there, wait for every source's tile when it receives, then advance
    /// the tile's epoch.
    fn ctas(&self, words: &BTreeMap<Word, u32>) -> Vec<Vec<Op>> {
        match self {
            Sim::Gather {
                outputs,
                flags,
                epoch,
                elements,
                rank,
                target,
                writer,
                generation,
            } => {
                let (rank, ranks, target) = (*rank, outputs.len() as u64, *target);
                let delivers = |dest: u64| target == ranks || dest == target;
                let tiles = elements.div_ceil(TILE_WORDS);
                (0..tiles)
                    .map(|tile| {
                        let cell = tile_word(*epoch, tile);
                        let next = words.get(&cell).copied().unwrap_or(0) + 1;
                        let flag = rank * tiles + tile;
                        let own = flags[rank as usize];
                        let mut ops = Vec::new();
                        for source in 0..ranks {
                            let ack = ranks * tiles + source * tiles + tile;
                            ops.push(Op::Release(tile_word(own, ack), next - 1));
                        }
                        for (dest, &flags) in flags.iter().enumerate() {
                            if delivers(dest as u64) {
                                ops.push(Op::Wait(
                                    tile_word(flags, ranks * tiles + flag),
                                    next - 1,
                                ));
                            }
                        }
                        for (dest, (output, &flags)) in outputs.iter().zip(flags).enumerate() {
                            if delivers(dest as u64) {
                                let first = rank * elements + tile * TILE_WORDS;
                                ops.push(Op::Store(
                                    tile_word(*output, first),
                                    *generation,
                                    *writer,
                                ));
                                ops.push(Op::Release(tile_word(flags, flag), next));
                            }
                        }
                        if delivers(rank) {
                            for source in 0..ranks {
                                ops.push(Op::Wait(tile_word(own, source * tiles + tile), next));
                            }
                        }
                        ops.push(Op::Release(cell, next));
                        ops
                    })
                    .collect()
            }
            Sim::Phased {
                outputs,
                flags,
                epoch,
                elements,
                rank,
                writer,
                generation,
                acknowledged,
            } => {
                // CTA 0's thread 0 does every flag step; the cluster's
                // stores of a destination fall between its acknowledgement
                // wait and its data flag.
                let tagged = |epoch: u32| 0x8000_0000 | epoch;
                let (rank, ranks) = (*rank, outputs.len() as u64);
                let cell = tile_word(*epoch, 0);
                let next = words.get(&cell).copied().unwrap_or(0) + 1;
                let own = flags[rank as usize];
                let mut ops = Vec::new();
                for source in (0..ranks).filter(|&source| source != rank) {
                    ops.push(Op::Release(
                        tile_word(flags[source as usize], ranks + rank),
                        tagged(next - 1),
                    ));
                }
                for phase in 0..ranks {
                    let dest = (rank + phase) % ranks;
                    if dest != rank && *acknowledged {
                        ops.push(Op::Wait(tile_word(own, ranks + dest), tagged(next - 1)));
                    }
                    for tile in 0..elements.div_ceil(TILE_WORDS) {
                        let first = rank * elements + tile * TILE_WORDS;
                        ops.push(Op::Store(
                            tile_word(outputs[dest as usize], first),
                            *generation,
                            *writer,
                        ));
                    }
                    if dest != rank {
                        ops.push(Op::Release(
                            tile_word(flags[dest as usize], rank),
                            tagged(next),
                        ));
                    }
                }
                for source in (0..ranks).filter(|&source| source != rank) {
                    ops.push(Op::Wait(tile_word(own, source), tagged(next)));
                }
                ops.push(Op::Release(cell, next));
                vec![ops]
            }
            Sim::Reduce { key, members } => {
                vec![vec![Op::Arrive(*key), Op::Barrier(*key, *members)]]
            }
            Sim::Consume { .. } => Vec::new(),
        }
    }

    /// A consumer's check against the stored tiles.
    fn consume(&self, stored: &BTreeMap<Word, (u32, u32)>) -> Result<(), String> {
        let Sim::Consume {
            output,
            elements,
            ranks,
            first,
            generation,
            symbol,
        } = self
        else {
            return Ok(());
        };
        for source in 0..*ranks {
            for tile in 0..elements.div_ceil(TILE_WORDS) {
                let word = tile_word(*output, source * elements + tile * TILE_WORDS);
                let found = stored.get(&word).copied();
                if found != Some((*generation, first + source as u32)) {
                    return Err(format!(
                        "{symbol} reads gather {generation} of rank {} but source {source} \
                         tile {tile} holds {found:?} (generation, writer)",
                        first + source as u32
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Which group boundaries a simulation keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Boundaries {
    /// The lowering's launches.
    All,
    /// Without the group barriers.
    NoBarrier,
    /// Without the barriers and the query gathers (and their reads): no
    /// group launch between one piece's merge and the next piece's
    /// exchange.
    None,
    /// As `None`, with the phased gathers' acknowledgement waits dropped.
    Unacknowledged,
}

/// Every rank's group launches of the indexer selection and the latent
/// attention of layer 3, over eager prefills split into `steps`' pieces
/// (first row, rows), in groups of 2, keeping `boundaries`: each
/// all-gather, each launch that reads what a gather delivered to the rank,
/// and an all-reduce barrier after each step.
fn group_launches(steps: &[&[(u32, u32)]], boundaries: Boundaries) -> Vec<Vec<Sim>> {
    let shape = Shape::TP4;
    let size = 2;
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| latent_format(format) == FORMATS[0] && !has_mtp_layer(format))
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<RankBuffers>>();
    let cp = group_state(&device, size);
    let ops = crate::ops::Ops::for_tests(shape, size);
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    let mut out = vec![Vec::new(); ranks.len()];
    // Every rank issues the same gathers, so its count names a call.
    let mut generations = vec![0; ranks.len()];
    for (index, pieces) in steps.iter().enumerate() {
        let rows = pieces.iter().map(|piece| piece.1).sum();
        let phase = Phase::Prefill { rows };
        for (rank, buffers) in ranks.iter().enumerate() {
            let inputs = inputs(rank as u32, phase);
            let pieces = pieces
                .iter()
                .map(|&(first, rows)| crate::lower::Piece {
                    first,
                    rows,
                    inputs,
                })
                .collect::<Vec<_>>();
            let lowering = Lowering {
                shape,
                format: &format,
                buffers,
                inputs: &inputs,
                ops: &ops,
                rank: rank as u32,
                index_topk: shape.profile.index_topk_max,
                pieces: &pieces,
                row_offset: 0,
                peers: &peers,
                head_rows: head_rows(phase),
                context_parallel: Some(&cp),
                draft: None,
                verify_tokens: None,
                projections: None,
            };
            let sims = &mut out[rank];
            let generation = &mut generations[rank];
            let first = rank as u32 / size * size;
            // Per output buffer, the last gather that delivered there.
            let mut delivered = BTreeMap::new();
            for stage in [Stage::IndexSelect(3), Stage::LatentAttention(3)] {
                for launch in lowering.lower(phase, stage).unwrap().unwrap() {
                    let arguments =
                        dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
                    if launch.symbol == PHASED_SYMBOL {
                        let KernelArgumentV1::DeviceBuffer(epoch) = arguments[4] else {
                            panic!("{:?}", arguments[4]);
                        };
                        let KernelArgumentV1::ScalarU64(elements) = arguments[5] else {
                            panic!("{:?}", arguments[5]);
                        };
                        let KernelArgumentV1::ScalarU32(group_rank) = arguments[6] else {
                            panic!("{:?}", arguments[6]);
                        };
                        let outputs = pointers(&arguments[2]);
                        let queries = outputs[group_rank as usize].buffer.slot()
                            == cp.ranks[rank].queries.buffer.slot();
                        let unordered =
                            matches!(boundaries, Boundaries::None | Boundaries::Unacknowledged);
                        if unordered && queries {
                            continue;
                        }
                        *generation += 1;
                        let output = outputs[group_rank as usize];
                        delivered.insert(output.buffer.slot(), (output, elements, *generation));
                        sims.push(Sim::Phased {
                            outputs,
                            flags: pointers(&arguments[3]),
                            epoch,
                            elements,
                            rank: group_rank.into(),
                            writer: rank as u32,
                            generation: *generation,
                            acknowledged: boundaries != Boundaries::Unacknowledged,
                        });
                    } else if launch.symbol == ALLGATHER_SYMBOL {
                        let KernelArgumentV1::DeviceBuffer(epoch) = arguments[3] else {
                            panic!("{:?}", arguments[3]);
                        };
                        let KernelArgumentV1::ScalarU64(elements) = arguments[4] else {
                            panic!("{:?}", arguments[4]);
                        };
                        let KernelArgumentV1::ScalarU32(group_rank) = arguments[5] else {
                            panic!("{:?}", arguments[5]);
                        };
                        let KernelArgumentV1::ScalarU32(target) = arguments[7] else {
                            panic!("{:?}", arguments[7]);
                        };
                        let outputs = pointers(&arguments[1]);
                        let queries = outputs[group_rank as usize].buffer.slot()
                            == cp.ranks[rank].queries.buffer.slot();
                        if (boundaries != Boundaries::All && elements == 1)
                            || (matches!(boundaries, Boundaries::None | Boundaries::Unacknowledged)
                                && queries)
                        {
                            continue;
                        }
                        *generation += 1;
                        if target == size || target == group_rank {
                            let output = outputs[group_rank as usize];
                            delivered.insert(output.buffer.slot(), (output, elements, *generation));
                        }
                        sims.push(Sim::Gather {
                            outputs,
                            flags: pointers(&arguments[2]),
                            epoch,
                            elements,
                            rank: group_rank.into(),
                            target: target.into(),
                            writer: rank as u32,
                            generation: *generation,
                        });
                    } else {
                        let read = arguments.iter().find_map(|argument| match argument {
                            KernelArgumentV1::DeviceBuffer(span) => {
                                delivered.get(&span.buffer.slot())
                            }
                            _ => None,
                        });
                        if let Some(&(output, elements, generation)) = read {
                            sims.push(Sim::Consume {
                                output,
                                elements,
                                ranks: size.into(),
                                first,
                                generation,
                                symbol: launch.symbol.clone(),
                            });
                        }
                    }
                }
            }
            sims.push(Sim::Reduce {
                key: (u32::MAX - 100, index as u64),
                members: ranks.len() as u32,
            });
        }
    }
    out
}

/// Runs every rank's launches in stream order with the words retained,
/// each launch's CTAs advancing as far as their waits allow. Rank
/// `delayed` runs a launch that reads a gather's output only when no other
/// step can advance (the slowest legal consumer); every other rank reads at
/// once. Returns the first stale read, and panics on a deadlock.
fn simulate(ranks: &[Vec<Sim>], delayed: usize) -> Result<(), String> {
    let mut words = BTreeMap::<Word, u32>::new();
    let mut stored = BTreeMap::<Word, (u32, u32)>::new();
    let mut at = vec![0; ranks.len()];
    let mut running: Vec<Option<Vec<std::collections::VecDeque<Op>>>> = vec![None; ranks.len()];
    loop {
        let mut progress = false;
        for rank in 0..ranks.len() {
            let Some(sim) = ranks[rank].get(at[rank]) else {
                continue;
            };
            if matches!(sim, Sim::Consume { .. }) {
                if rank != delayed {
                    sim.consume(&stored)
                        .map_err(|error| format!("rank {rank}: {error}"))?;
                    at[rank] += 1;
                    progress = true;
                }
                continue;
            }
            let ctas = running[rank].get_or_insert_with(|| {
                progress = true;
                sim.ctas(&words).into_iter().map(Into::into).collect()
            });
            for cta in ctas.iter_mut() {
                while let Some(&op) = cta.front() {
                    match op {
                        Op::Wait(word, value) | Op::Barrier(word, value) => {
                            if words.get(&word).copied().unwrap_or(0) != value {
                                break;
                            }
                        }
                        Op::Release(word, value) => {
                            words.insert(word, value);
                        }
                        Op::Store(word, generation, writer) => {
                            stored.insert(word, (generation, writer));
                        }
                        Op::Arrive(word) => *words.entry(word).or_default() += 1,
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
        if progress {
            continue;
        }
        // Nothing else advances: the first waiting consumer reads now.
        let Some(rank) = (0..ranks.len())
            .find(|&rank| matches!(ranks[rank].get(at[rank]), Some(Sim::Consume { .. })))
        else {
            assert!(
                at.iter().zip(ranks).all(|(&at, sims)| at == sims.len()),
                "deadlock at launches {at:?}"
            );
            return Ok(());
        };
        ranks[rank][at[rank]]
            .consume(&stored)
            .map_err(|error| format!("rank {rank}: {error}"))?;
        at[rank] += 1;
    }
}

/// A rank can enter a cold bucket's gather while a peer still reads what
/// another bucket's gather delivered: each bucket gathers pairs into its
/// own output, the phased gathers send to a member only once it entered
/// the same call (even a state's first), and the group barrier ends each
/// attention piece before any member's next exchange or query gather. A
/// 2,560-row prefill's pieces of 2,048 and 512 rows run their selection
/// and attention with one rank's consumers (the top-k merge, the partial
/// attention, its merge as owner) as late as the protocols allow, then a
/// 4,096-row prefill (one bucket twice) and the 2,560-row prefill again
/// (warm buckets): for every delayed rank, every consumer reads the gather
/// it follows, each slot written by its own group member. Without the
/// barriers and the query gathers the exchanges' acknowledgements still
/// order the received blocks; with those waits dropped as well, a delayed
/// owner's blocks are overwritten.
#[test]
fn mixed_bucket_pieces_never_overwrite_a_live_gather() {
    let cold: &[(u32, u32)] = &[(0, 2048), (2048, 512)];
    let same: &[(u32, u32)] = &[(0, 2048), (2048, 2048)];
    let ranks = group_launches(&[cold, same, cold], Boundaries::All);
    // Each piece gathers pairs and queries and exchanges the partial
    // blocks (phased), then passes the barrier.
    let count = |phased: bool| {
        ranks[1]
            .iter()
            .filter(|sim| match sim {
                Sim::Phased { .. } => phased,
                Sim::Gather { .. } => !phased,
                _ => false,
            })
            .count()
    };
    assert_eq!((count(true), count(false)), (3 * 2 * 3, 3 * 2));
    for consumer in ["topk_merge", "prefill_partial", "partial_merge"] {
        assert!(
            ranks[1]
                .iter()
                .any(|sim| matches!(sim, Sim::Consume { symbol, .. } if symbol.contains(consumer))),
            "{consumer}"
        );
    }
    for delayed in 0..4 {
        simulate(&ranks, delayed).unwrap();
    }
    // The exchange's acknowledgements order the received blocks without
    // the barriers and the query gathers; without them as well a delayed
    // owner's merge reads blocks of a later piece.
    for boundaries in [Boundaries::NoBarrier, Boundaries::None] {
        let unbarred = group_launches(&[cold, same, cold], boundaries);
        for delayed in 0..4 {
            simulate(&unbarred, delayed).unwrap();
        }
    }
    let unordered = group_launches(&[cold, same, cold], Boundaries::Unacknowledged);
    let errors = (0..4)
        .filter_map(|delayed| simulate(&unordered, delayed).err())
        .collect::<Vec<_>>();
    assert!(!errors.is_empty());
    assert!(
        errors.iter().all(|error| error.contains("partial_merge")),
        "{errors:#?}"
    );
}

/// An instance served at context-parallel size `size` on `host`.
fn create_group(
    host: &std::rc::Rc<crate::fake_host_tests::Host>,
    size: u32,
) -> runtime_bundle_abi::BundleInstanceV1 {
    let mut tail = runtime_bundle_sdk::context_parallel::TAG.to_vec();
    tail.extend_from_slice(&size.to_le_bytes());
    tail.extend(crate::config::tests::options(2048));
    let record =
        crate::config::tests::record(model_glm5next_contract::contract::DECODE_BATCH, &tail);
    crate::fake_host_tests::api()
        .create(HostServicesTableV1::new(std::rc::Rc::clone(host)), &record)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()))
}

/// The span of snapshot `piece` in a rank's request-state `buffers`.
fn piece_span(
    buffers: &[DeviceBufferSpanV1],
    piece: &bundle_runtime::snapshot::Descriptor,
) -> DeviceBufferSpanV1 {
    let view = buffers[piece.buffer];
    assert!(!piece.spare && piece.offset + piece.length <= view.length);
    DeviceBufferSpanV1::new(view.buffer, view.offset + piece.offset, piece.length)
}

/// Writes distinct bytes (from `seed` and the rank) into every declared
/// piece of `slot`'s committed state on every rank, and returns them per
/// rank in snapshot order.
pub(crate) fn fill_slot(
    host: &crate::fake_host_tests::FakeHost,
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
    seed: u64,
) -> Vec<Vec<u8>> {
    let (spans, table) = crate::instance::snapshot_layout(handle, slot);
    spans
        .iter()
        .enumerate()
        .map(|(rank, buffers)| {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed << 40 ^ (rank as u64) << 56;
            let mut all = Vec::new();
            for piece in &table.descriptors {
                let bytes = (0..piece.length.div_ceil(8))
                    .flat_map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        state.to_le_bytes()
                    })
                    .take(piece.length as usize)
                    .collect::<Vec<_>>();
                host.write(piece_span(buffers, piece), &bytes);
                all.extend(bytes);
            }
            all
        })
        .collect()
}

/// Every rank's bytes of `slot`'s declared pieces, in snapshot order.
pub(crate) fn slot_bytes(
    host: &crate::fake_host_tests::FakeHost,
    handle: runtime_bundle_abi::BundleHandleV1,
    slot: u32,
) -> Vec<Vec<u8>> {
    let (spans, table) = crate::instance::snapshot_layout(handle, slot);
    spans
        .iter()
        .map(|buffers| {
            table
                .descriptors
                .iter()
                .flat_map(|piece| host.read(piece_span(buffers, piece)))
                .collect()
        })
        .collect()
}

/// The snapshot bytes rank `rank` holds in its own local pages `pages`
/// (the stripes' indexes), header included.
fn snapshot_bytes(
    host: &crate::fake_host_tests::FakeHost,
    table: &bundle_runtime::snapshot::Table,
    rank: u32,
    pages: &[u64],
) -> Vec<u8> {
    let mut bytes = pages
        .iter()
        .flat_map(|&page| {
            host.read(DeviceBufferSpanV1::new(
                runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(
                    1,
                    crate::fake_host_tests::KV_SLOT + rank,
                ),
                page * table.page_bytes,
                table.page_bytes,
            ))
        })
        .collect::<Vec<_>>();
    bytes.truncate((table.header_at() + table.header_bytes) as usize);
    bytes
}

/// The journal lines from `from` on that start with `prefix`.
fn count(host: &crate::fake_host_tests::FakeHost, from: usize, prefix: &str) -> usize {
    host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with(prefix))
        .count()
}

/// At context-parallel sizes 2 and 4 a snapshot's page ids are stripes:
/// every rank keeps the stripe's local page in its own pool, and the
/// request state (KDA state and the per-item indexer tail) is each rank's
/// own, as at size 1. A capture puts each rank's committed pieces, and no
/// other rank's, into that rank's pages at the snapshot offsets (rank 0
/// also the header); a restore into another slot takes each rank's pages,
/// not the source slot, and leaves a third slot unwritten. A copy that
/// fails partway through rank 1 fails the capture or the restore with
/// every attempted rank drained: the source keeps its owner, the
/// destination stays free and a retry succeeds.
#[test]
fn striped_prefix_snapshots_move_each_ranks_bytes_from_its_origin() {
    for size in [2, 4] {
        let host = crate::fake_host_tests::FakeHost::new((3, 32));
        let mut instance = create_group(&host, size);
        let handle = instance.handle();
        crate::instance::seed(handle, 7, 2, 300);
        host.backing.set(true);
        let source = fill_slot(&host, handle, 2, u64::from(size));
        // Every rank's bytes differ, so a rank holding another's shard
        // shows.
        assert!((1..4).all(|rank| source[rank] != source[0]));
        let (_, table) = crate::instance::snapshot_layout(handle, 2);
        let held = table.pages();
        assert_eq!(held, instance.describe_prefix_state().unwrap());
        // Stripes out of order from 100 on, a stride prime to their count.
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let stride = (11..).find(|&stride| gcd(stride, held) == 1).unwrap();
        let pages = (0..held)
            .map(|index| 100 + (index * stride) % held)
            .collect::<Vec<_>>();
        let start = host.device.journal.borrow().len();
        instance.capture_prefix_state(7, 300, &pages).unwrap();
        let copies = count(&host, start, "copy ");
        assert_eq!(copies % 4, 0);
        let check_pages = |expected: &[Vec<u8>]| {
            for rank in 0..4 {
                let bytes = snapshot_bytes(&host, &table, rank, &pages);
                for piece in &table.descriptors {
                    let at = piece.at as usize..(piece.at + piece.length) as usize;
                    assert!(
                        bytes[at.clone()] == expected[rank as usize][at],
                        "size {size} rank {rank} piece {piece:?}"
                    );
                }
                let header = &bytes[table.header_at() as usize..];
                assert_eq!(header.iter().any(|&byte| byte != 0), rank == 0);
            }
        };
        check_pages(&source);
        // The source slot changes; slot 6 restores the pages' bytes.
        let changed = fill_slot(&host, handle, 2, u64::from(size) + 10);
        let tokens = (0..300).collect::<Vec<u32>>();
        instance
            .restore_prefix_state_in_slot(9, &tokens, &pages, 6)
            .unwrap();
        assert!(slot_bytes(&host, handle, 6) == source, "size {size}");
        assert!(slot_bytes(&host, handle, 2) == changed);
        assert!(
            slot_bytes(&host, handle, 3)
                .iter()
                .flatten()
                .all(|&byte| byte == 0)
        );
        // A capture whose copies fail in rank 1 drains the attempted ranks;
        // request 7 keeps slot 2 and a retry captures its new bytes.
        host.fail_copy.set(Some(copies as u32 / 4 + 1));
        let start = host.device.journal.borrow().len();
        assert_eq!(
            instance.capture_prefix_state(7, 300, &pages).unwrap_err(),
            runtime_bundle_abi::BundleAbiError::Status(BundleStatusV1::HostError)
        );
        assert!(count(&host, start, "synchronize ") >= 2);
        assert_eq!(
            crate::instance::ledger_state(handle, 2),
            (Some((7, 300)), false)
        );
        instance.capture_prefix_state(7, 300, &pages).unwrap();
        check_pages(&changed);
        // A restore whose copies fail leaves slot 7 free; a retry fills it.
        host.fail_copy.set(Some(copies as u32 / 4 + 1));
        assert_eq!(
            instance
                .restore_prefix_state_in_slot(11, &tokens, &pages, 7)
                .unwrap_err(),
            runtime_bundle_abi::BundleAbiError::Status(BundleStatusV1::HostError)
        );
        assert_eq!(crate::instance::ledger_state(handle, 7), (None, false));
        instance
            .restore_prefix_state_in_slot(11, &tokens, &pages, 7)
            .unwrap();
        assert!(slot_bytes(&host, handle, 7) == changed);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
