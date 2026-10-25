use super::*;
use bundle_runtime::pipeline::{self, Collectives, PreparedLaunch, Unit};
use runtime_bundle_abi::{HostServicesBackendV1, HostServicesTableV1};
use std::collections::BTreeMap;
use std::rc::Rc;

fn argument_span(launch: &Lowered, index: usize) -> Span {
    let kernel_core::contract::Argument::DeviceBuffer(id) = launch.launch.arguments.0[index] else {
        panic!("expected a bound device buffer");
    };
    launch.launch.bindings.get(id).unwrap()
}

#[test]
fn every_topology_dispatches_all_five_head_positions_on_the_fake_host() {
    for shape in crate::shape::topologies() {
        for mode in [DraftMode::Greedy, DraftMode::Sampled] {
            dispatch_positions(shape, mode);
        }
    }
}

const NOISE: &str = crate::sm120::dspark_noise::SYMBOL;

fn dispatch_positions(shape: Shape, mode: DraftMode) {
    {
        let kernels = crate::Kernels::new(&shape).unwrap();
        let heads =
            Heads::prepare(kernels.backend.as_ref(), kernels.own.as_ref(), shape, mode).unwrap();
        let host = crate::fake_host_tests::FakeHost::new(shape, crate::config::KV_LAYOUTS[0]);
        let table = HostServicesTableV1::new(Rc::clone(&host));
        let services = table.caller();
        let items = 3_u32;
        let rows = items * 5;
        let dim = u64::from(shape.geometry.shape.dspark.markov_rank);
        let hidden = u64::from(shape.geometry.shape.hidden);
        let vocab = u64::from(shape.vocab_rows());
        let mut peers = Vec::new();
        let mut views = Vec::new();
        let mut streams = Vec::new();
        let mut symbols = Vec::new();
        let mut programs = Vec::new();
        let mut position_rows = Vec::new();
        for rank in 0..shape.ranks() {
            let context = host.select_context(rank, rank as i32).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            streams.push(stream);
            let alloc = |bytes| {
                Span::new(
                    host.allocate_device_buffer(pool, stream, bytes, 256)
                        .unwrap(),
                    0,
                    bytes,
                )
            };
            let mut names = BTreeMap::new();
            for cubin in crate::CUBINS
                .iter()
                .filter(|c| crate::serves(c.name, &shape))
            {
                let module = services.load_cuda_module(context, cubin.bytes).unwrap();
                for symbol in cubin.symbols {
                    names.insert(
                        (*symbol).to_owned(),
                        services
                            .resolve_cuda_symbol(module, symbol.as_bytes())
                            .unwrap(),
                    );
                }
            }
            symbols.push(names);
            let (reduce, gather) = heads.state();
            let embedding = alloc(u64::from(items) * dim * 2);
            let embedding_weight = alloc(vocab * dim * 2);
            let gathered = alloc(u64::from(DECODE_BATCH * 2 * shape.ranks()) * 4);
            peers.push(Peer {
                embedding_weight,
                embedding,
                reduce_state: reduce.iter().map(|&n| alloc(n)).collect(),
                gathered,
                gather_state: gather.iter().map(|&n| alloc(n)).collect(),
            });
            let hidden_f32 = alloc(u64::from(rows) * hidden * 4);
            let block_logits = alloc(u64::from(rows) * vocab * 4);
            let head = HeadViews {
                streams: alloc(
                    u64::from(rows) * hidden * u64::from(shape.geometry.shape.hc.streams) * 2,
                ),
                pre: alloc(u64::from(rows) * u64::from(shape.geometry.shape.hc.streams) * 4),
                hidden_f32,
                norm_weight: alloc(hidden * 2),
                normalized_f32: alloc(u64::from(rows) * hidden * 4),
                vocabulary_weight: alloc(vocab * hidden * 2),
                projection_state: crate::ops::exact_f32::ProjectionState {
                    weight_domain: Some(alloc(vocab.div_ceil(16) * 4)),
                    workspace: alloc(heads.vocabulary.info().workspace),
                },
                logits: block_logits,
                live_rows: alloc(4),
                status: alloc(u64::from(rows) * 4),
            };
            services
                .copy_host_to_device(stream, head.live_rows, &rows.to_le_bytes())
                .unwrap();
            services
                .copy_host_to_device(stream, head.status, &vec![0; head.status.length as usize])
                .unwrap();
            let program = heads.vocabulary(rows, &head).unwrap();
            assert!(program[0].launch.symbol.ends_with("weighted_head_f32_v1"));
            assert_eq!(argument_span(&program[0], 0), head.streams);
            assert_eq!(argument_span(&program[0], 1), head.pre);
            assert_eq!(argument_span(&program[0], 2), head.hidden_f32);
            assert_eq!(argument_span(&program[1], 0), head.hidden_f32);
            assert!(program[0].launch.plan.pdl);
            let short = HeadViews {
                hidden_f32: Span::new(head.hidden_f32.buffer, 0, head.hidden_f32.length / 2),
                ..head
            };
            assert!(
                heads.vocabulary(rows, &short).is_err(),
                "BF16-sized collapse output must be refused"
            );
            for alias in [head.streams, head.pre, head.normalized_f32, head.logits] {
                let bad = HeadViews {
                    hidden_f32: alias,
                    ..head
                };
                assert_eq!(
                    heads.vocabulary(rows, &bad).err().unwrap().0,
                    "DSpark collapsed hidden must have separate storage"
                );
            }
            programs.push(program);
            views.push(PositionViews {
                predecessor_ids: alloc(u64::from(items) * 4),
                embedding_weight,
                embedding_partial: alloc(u64::from(items) * dim * 2),
                embedding_wide: alloc(u64::from(items) * dim * 4),
                embedding,
                embedding_copy: alloc(u64::from(items) * dim * 2),
                block_logits,
                position_rows: alloc(u64::from(items) * 4),
                position_logits: alloc(u64::from(items) * vocab * 4),
                markov_weight: alloc(vocab * dim * 2),
                hidden_f32,
                confidence_weight: alloc((hidden + dim) * 2),
                confidence: alloc(u64::from(rows) * 4),
                pairs: alloc(u64::from(DECODE_BATCH * 2) * 4),
                gathered,
                live_width: alloc(4),
                live_items: alloc(4),
                noise: alloc(u64::from(DECODE_BATCH) * 32),
                draft_ids: alloc(u64::from(rows) * 4),
            });
            let view = views.last().unwrap();
            services
                .copy_host_to_device(stream, view.live_items, &items.to_le_bytes())
                .unwrap();
            services
                .copy_host_to_device(stream, view.live_width, &shape.vocab_rows().to_le_bytes())
                .unwrap();
            services
                .copy_host_to_device(
                    stream,
                    view.predecessor_ids,
                    &(0..items)
                        .flat_map(|_| 42_u32.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            services
                .copy_host_to_device(stream, view.pairs, &vec![0; view.pairs.length as usize])
                .unwrap();
            for &state in peers
                .last()
                .unwrap()
                .reduce_state
                .iter()
                .chain(&peers.last().unwrap().gather_state)
            {
                services
                    .copy_host_to_device(stream, state, &vec![0; state.length as usize])
                    .unwrap();
            }
            position_rows.push(
                (0..5)
                    .map(|position| {
                        let span = alloc(u64::from(items) * 4);
                        let bytes = (0..items)
                            .flat_map(|item| (item * 5 + position).to_le_bytes())
                            .collect::<Vec<_>>();
                        services.copy_host_to_device(stream, span, &bytes).unwrap();
                        span
                    })
                    .collect::<Vec<_>>(),
            );
        }
        for rank in 0..shape.ranks() {
            for position in 0..5 {
                let v = PositionViews {
                    position_rows: position_rows[rank as usize][position as usize],
                    ..views[rank as usize]
                };
                let program = heads.position(rank, position, items, &v, &peers).unwrap();
                if position < 3 {
                    let packed = heads
                        .position_packed(rank, position, items, &v, &peers, true)
                        .unwrap();
                    let ordinary: Vec<_> = program
                        .iter()
                        .filter(|op| op.launch.symbol != "chop_row_gather_by_index_v1")
                        .collect();
                    assert_eq!(packed.len(), ordinary.len());
                    for (old, new) in ordinary.iter().zip(&packed) {
                        assert_eq!(old.launch.symbol, new.launch.symbol);
                        assert_eq!(old.launch.plan, new.launch.plan);
                        assert_eq!(old.collective, new.collective);
                        // Only the Markov base changes from a gathered position
                        // to its contiguous position-major slice. Every other
                        // scalar, view, confidence and Gumbel operand is exact.
                        for (index, (a, b)) in old
                            .launch
                            .arguments
                            .0
                            .iter()
                            .zip(&new.launch.arguments.0)
                            .enumerate()
                        {
                            if old.launch.symbol == "chop_deepseek41_dspark_markov_tiled_v1"
                                && index == 2
                            {
                                continue;
                            }
                            match (a, b) {
                                (
                                    kernel_core::contract::Argument::DeviceBuffer(a),
                                    kernel_core::contract::Argument::DeviceBuffer(b),
                                ) => assert_eq!(
                                    old.launch.bindings.get(*a).unwrap(),
                                    new.launch.bindings.get(*b).unwrap()
                                ),
                                _ => assert_eq!(a, b),
                            }
                        }
                    }
                    let step = packed
                        .iter()
                        .find(|op| op.launch.symbol == "chop_deepseek41_dspark_markov_tiled_v1")
                        .unwrap();
                    assert_eq!(
                        argument_span(step, 2),
                        slice(
                            v.block_logits,
                            u64::from(position * items) * vocab * 4,
                            u64::from(items) * vocab * 4
                        )
                        .unwrap()
                    );
                }
                // Sampled mode adds the noise between the Markov step and
                // the argmax, on the Markov output, at this position and
                // this rank's first global ID; greedy mode has none.
                let at = |symbol: &str| program.iter().position(|op| op.launch.symbol == symbol);
                let markov_at = at("chop_deepseek41_dspark_markov_tiled_v1").unwrap();
                let argmax_at = at("chop_dspark_argmax_f32_wide_v1").unwrap();
                match (mode, at(NOISE)) {
                    (DraftMode::Greedy, noise) => assert_eq!(noise, None),
                    (DraftMode::Sampled, Some(noise_at)) => {
                        assert!(markov_at < noise_at && noise_at < argmax_at);
                        let noise = &program[noise_at];
                        assert_eq!(argument_span(noise, 0), v.position_logits);
                        assert_eq!(argument_span(noise, 1), v.noise);
                        assert_eq!(argument_span(noise, 2), v.live_items);
                        assert_eq!(argument_span(noise, 3), v.live_width);
                        assert_eq!(
                            noise.launch.arguments.0[4..],
                            [
                                kernel_core::contract::Argument::U32(shape.vocab_rows()),
                                kernel_core::contract::Argument::U32(rank * shape.vocab_rows()),
                                kernel_core::contract::Argument::U32(position),
                            ]
                        );
                        assert_eq!(
                            noise.launch.plan.grid,
                            [items, shape.vocab_rows().div_ceil(2048), 1]
                        );
                        assert!(!noise.collective);
                        assert_eq!(argument_span(&program[argmax_at], 0), v.position_logits);
                    }
                    (DraftMode::Sampled, None) => panic!("sampled mode adds the noise"),
                }
                let confidence = program
                    .iter()
                    .find(|op| op.launch.symbol == "chop_deepseek41_dspark_confidence_parallel_v1")
                    .unwrap();
                assert_eq!(confidence.launch.plan.grid, [items, 1, 1]);
                // One weight tile serves every row; rows stay off the grid.
                let markov = program
                    .iter()
                    .find(|op| op.launch.symbol == "chop_deepseek41_dspark_markov_tiled_v1")
                    .unwrap();
                assert_eq!(
                    markov.launch.plan.grid,
                    [shape.vocab_rows().div_ceil(128), 1, 1]
                );
                // One CTA per row reduces the rank's shard with 1024 threads.
                let argmax = program
                    .iter()
                    .find(|op| op.launch.symbol == "chop_dspark_argmax_f32_wide_v1")
                    .unwrap();
                assert_eq!(
                    (argmax.launch.plan.grid, argmax.launch.plan.block),
                    ([items, 1, 1], [1024, 1, 1])
                );
                // One launch reads each owner's shard through every rank's
                // table, with no cross-rank wait.
                let lookup = program
                    .iter()
                    .find(|op| op.launch.symbol == peer_symbol(&shape))
                    .unwrap();
                assert_eq!(lookup.launch.plan.grid, [items, 1, 1]);
                assert!(!lookup.collective);
                let kernel_core::contract::Argument::DevicePointerArray(tables) =
                    &lookup.launch.arguments.0[0]
                else {
                    panic!("expected the peer tables");
                };
                let tables = tables
                    .iter()
                    .map(|&id| lookup.launch.bindings.get(id).unwrap())
                    .collect::<Vec<_>>();
                let weights = peers.iter().map(|p| p.embedding_weight).collect::<Vec<_>>();
                assert_eq!(tables, weights);
                assert!(
                    program.iter().all(
                        |op| !op.collective || op.launch.symbol == "chop_comm_allgather_f32_v1"
                    )
                );
                assert!(
                    program
                        .iter()
                        .any(|op| op.launch.symbol == "chop_row_gather_by_index_v1")
                );
                // A lone item's step reads its block row in place.
                let lone = heads.position(rank, position, 1, &v, &peers).unwrap();
                assert!(
                    lone.iter()
                        .all(|op| op.launch.symbol != "chop_row_gather_by_index_v1")
                );
                let step = lone
                    .iter()
                    .find(|op| op.launch.symbol == "chop_deepseek41_dspark_markov_row_v1")
                    .unwrap();
                assert_eq!(argument_span(step, 3), v.position_logits);
                let row = vocab * 4;
                assert_eq!(
                    argument_span(step, 2),
                    Span::new(
                        v.block_logits.buffer,
                        v.block_logits.offset + u64::from(position) * row,
                        row
                    )
                );
                let hidden = argument_span(confidence, 0);
                assert_eq!(hidden.buffer, v.hidden_f32.buffer);
                assert_eq!(
                    hidden.offset,
                    v.hidden_f32.offset + u64::from(position) * hidden_width(&shape)
                );
                assert_ne!(hidden.buffer, v.block_logits.buffer);
                programs[rank as usize].extend(program);
            }
        }
        let (programs, waits): (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>) = programs
            .into_iter()
            .map(|program| {
                program
                    .into_iter()
                    .map(|l| (l.launch, l.collective))
                    .unzip()
            })
            .unzip();
        pipeline::check_collectives(&programs, &waits).unwrap();
        let segments = programs
            .iter()
            .zip(&waits)
            .map(|(p, w)| pipeline::collective_segments(p, w))
            .collect::<Vec<_>>();
        let collectives = Collectives::default();
        pipeline::launch_segments(
            &collectives,
            &mut Unit::default(),
            &mut vec![false; shape.ranks() as usize],
            &segments,
            |rank, launches| {
                pipeline::launch_all(services, streams[rank], &symbols[rank], launches)
            },
            |_| {},
            |_| Ok(()),
        )
        .unwrap();
        let journal = host.journal.borrow();
        assert_eq!(
            journal
                .iter()
                .filter(|line| line.starts_with("launch") && line.contains(NOISE))
                .count(),
            if mode == DraftMode::Sampled {
                shape.ranks() as usize * 5
            } else {
                0
            }
        );
        for symbol in [
            peer_symbol(&shape).as_str(),
            "chop_deepseek41_dspark_markov_tiled_v1",
            "chop_deepseek41_dspark_confidence_parallel_v1",
            "chop_dspark_argmax_f32_wide_v1",
            "chop_dspark_argmax_final_v1",
        ] {
            assert_eq!(
                journal
                    .iter()
                    .filter(|line| line.starts_with("launch") && line.contains(symbol))
                    .count(),
                shape.ranks() as usize * 5,
                "{} {symbol}",
                shape.topology()
            );
        }
        assert_eq!(
            journal
                .iter()
                .filter(|line| line.starts_with("launch") && line.contains("weighted_head_f32_v1"))
                .count(),
            shape.ranks() as usize
        );
        eprintln!("DSpark head journal: {} {mode:?}", shape.topology());
    }
}

fn peer_symbol(shape: &Shape) -> String {
    format!(
        "chop_{}_dspark_markov_replicated_tp{}_v1",
        shape.geometry.kernel_tag,
        shape.ranks()
    )
}

fn hidden_width(shape: &Shape) -> u64 {
    u64::from(shape.geometry.shape.hidden) * 4
}
