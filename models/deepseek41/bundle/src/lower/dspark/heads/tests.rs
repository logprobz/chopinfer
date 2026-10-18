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
        let kernels = crate::Kernels::new(&shape).unwrap();
        let heads = Heads::prepare(
            kernels.backend.as_ref(),
            kernels.own.as_ref(),
            shape,
            DraftMode::Greedy,
        )
        .unwrap();
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
                .filter(|c| crate::serves(c.name, shape.geometry.id))
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
            let gathered = alloc(u64::from(DECODE_BATCH * 2 * shape.ranks()) * 4);
            peers.push(Peer {
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
                embedding_weight: alloc(vocab * dim * 2),
                embedding_partial: alloc(u64::from(items) * dim * 2),
                embedding_wide: alloc(u64::from(items) * dim * 4),
                embedding,
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
                let confidence = program
                    .iter()
                    .find(|op| op.launch.symbol == "chop_deepseek41_dspark_confidence_v1")
                    .unwrap();
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
        for symbol in [
            "chop_deepseek41_dspark_markov_v1",
            "chop_deepseek41_dspark_confidence_v1",
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
        eprintln!("DSpark head journal: {}", shape.topology());
    }
}

fn hidden_width(shape: &Shape) -> u64 {
    u64::from(shape.geometry.shape.hidden) * 4
}
