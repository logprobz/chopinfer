//! Uses the bundle's fake package so the complete block goes through the
//! same ABI dispatch and collective segmentation as the target program.
use super::*;
use crate::buffers::{RankBuffers, Scratch, StateSpans};
use crate::dspark::execution::DraftInput;
use crate::dspark::state::DraftBlock;
use crate::lower::dspark::{self, BlockViews, CatchUpViews, DraftMode, Prepared, Window};
use crate::lower::{KvExtent, Lowering};
use crate::ops::{Lowered, Ops};
use crate::program::Phase;
use bundle_runtime::pipeline::{self, Collectives, PreparedLaunch, Unit};
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use model_deepseek41_contract::serving::{DeepSeek, PREFILL_TOKENS};
use std::collections::BTreeMap;

fn launch(
    services: &HostServicesV1,
    streams: &[BundleStreamHandleV1],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    programs: Vec<Vec<Lowered>>,
) {
    let (programs, waits): (Vec<Vec<PreparedLaunch>>, Vec<Vec<bool>>) = programs
        .into_iter()
        .map(|p| p.into_iter().map(|l| (l.launch, l.collective)).unzip())
        .unzip();
    pipeline::check_collectives(&programs, &waits).unwrap();
    let segments = programs
        .iter()
        .zip(&waits)
        .map(|(p, w)| pipeline::collective_segments(p, w))
        .collect::<Vec<_>>();
    pipeline::launch_segments(
        &Collectives::default(),
        &mut Unit::default(),
        &mut vec![false; streams.len()],
        &segments,
        |rank, p| pipeline::launch_all(services, streams[rank], &symbols[rank], p),
        |_| {},
        |_| Ok(()),
    )
    .unwrap();
}

#[test]
fn dspark_full_block_journal_at_each_topology() {
    for shape in crate::shape::topologies() {
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = Ops::prepare_set(
            &kernels,
            &shape,
            KvStorage::Native,
            model_deepseek41_contract::moe::Experts {
                count: shape.geometry.shape.dspark.experts,
                top_k: shape.geometry.shape.dspark.top_k,
            },
        );
        assert!(ops.glue.is_some(), "{:?}", ops.missing);
        let draft = Prepared::prepare(kernels.backend.as_ref(), shape, DraftMode::Greedy).unwrap();
        let mut shared = ops.shared_bytes();
        for (symbol, bytes) in draft.symbols() {
            let most = shared.entry(symbol.clone()).or_insert(0);
            *most = (*most).max(*bytes);
        }
        let mut held = Vec::new();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let table = HostServicesTableV1::new(Rc::clone(&host));
        let services = table.caller();
        let g = shape.geometry;
        let extent = Extent {
            served: 4,
            bucket: 4,
            verify_rows: 6,
        };
        let declaration =
            model_deepseek41_contract::state::request_state(g, KvStorage::Native, true);
        let state_bytes = declaration.buffers.iter().map(|b| b.bytes(extent)).sum();
        let mut buffers = Vec::new();
        let mut streams = Vec::new();
        let mut pools = Vec::new();
        let mut symbols = Vec::new();
        let mut inputs = Vec::new();
        for rank in 0..shape.ranks() {
            let context = host.select_context(rank, rank as i32).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            streams.push(stream);
            pools.push(pool);
            let alloc = |bytes| {
                DeviceBufferSpanV1::new(
                    host.allocate_device_buffer(pool, stream, bytes, 256)
                        .unwrap(),
                    0,
                    bytes,
                )
            };
            let scratch =
                bundle_runtime::scratch::bind(Scratch::of(&shape), |bytes| Ok(alloc(bytes)))
                    .unwrap();
            let pre = (0..crate::buffers::scratch_rows())
                .flat_map(|_| {
                    model_deepseek41_contract::hc::first_pre(g)
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                })
                .collect::<Vec<_>>();
            services
                .copy_host_to_device(stream, scratch.first_pre, &pre)
                .unwrap();
            services
                .zero_device_buffer(stream, scratch.row_status)
                .unwrap();
            services
                .zero_device_buffer(stream, scratch.shared_row_status)
                .unwrap();
            let mut weights = BTreeMap::new();
            for (name, bytes, _) in super::views(&shape) {
                let span = alloc(bytes);
                let flags = ops
                    .exact_f32
                    .as_ref()
                    .and_then(|dense| dense.weight_op(&name))
                    .and_then(|op| op.info().bytes_of("scratch.weight_domain"))
                    .map(alloc);
                if let Some(suffix) = name.strip_prefix("layers.0.") {
                    for stage in 0..g.dspark_stages() {
                        weights.insert(format!("mtp.{stage}.{suffix}"), span);
                        if let Some(flags) = flags {
                            weights.insert(
                                crate::ops::exact_f32::domain_name(&format!(
                                    "mtp.{stage}.{suffix}"
                                )),
                                flags,
                            );
                        }
                    }
                }
                if let Some(flags) = flags {
                    weights.insert(crate::ops::exact_f32::domain_name(&name), flags);
                }
                weights.insert(name, span);
            }
            let k = u64::from(g.shape.hidden) * g.shape.dspark.targets.len() as u64;
            weights.insert(
                "mtp.0.main_proj.weight".into(),
                alloc(u64::from(g.shape.hidden) * k),
            );
            weights.insert(
                "mtp.0.main_proj.scale".into(),
                alloc(u64::from(g.shape.hidden) * k / 32),
            );
            weights.insert(
                "mtp.0.main_norm.weight".into(),
                alloc(u64::from(g.shape.hidden) * 2),
            );
            let comm = ops
                .reduce
                .as_ref()
                .unwrap()
                .state()
                .iter()
                .map(|&bytes| {
                    let span = alloc(bytes);
                    services.zero_device_buffer(stream, span).unwrap();
                    span
                })
                .collect();
            let mix = ops.hc.as_ref().unwrap().scratch();
            // Each entry loads with the dynamic shared memory its op states,
            // the drafter's ops' as well as the instance's, as create loads
            // the program's.
            let mut names = BTreeMap::new();
            for cubin in crate::CUBINS
                .iter()
                .filter(|c| crate::serves(c.name, &shape))
            {
                bundle_runtime::acquire::load_module(
                    bundle_runtime::acquire::Log {
                        services,
                        bundle: "DeepSeek V4.1 drafter journal",
                    },
                    context,
                    (cubin.name, cubin.bytes),
                    cubin.symbols.iter().copied(),
                    |symbol| Ok(shared.get(symbol).copied().unwrap_or(0)),
                    &mut names,
                    &mut held,
                )
                .unwrap();
            }
            symbols.push(names);
            let layout = InputLayout::new::<DeepSeek>(PREFILL_TOKENS, PREFILL_TOKENS);
            inputs.push(DeviceInputs::within(
                PhaseArena {
                    arena: alloc(layout.end),
                    block_tables: alloc(4096),
                },
                &layout,
            ));
            let state_view = alloc(state_bytes);
            buffers.push(RankBuffers {
                scratch,
                state: StateSpans::new(&declaration, extent, state_view).unwrap(),
                views: weights,
                kv_pool: alloc(4096),
                comm,
                logits: None,
                sorted_rows: alloc(ops.moe.as_ref().unwrap().sorted_rows_bytes()),
                workspace: alloc(ops.workspace().max(draft.workspace()).max(4)),
                shared_workspace: alloc(ops.shared_workspace().max(4)),
                prefill_cache: None,
                mix_partials: mix.partials.map(alloc),
                mix_tickets: mix.tickets.map(alloc),
                rope_compressed: alloc(
                    u64::from(model_deepseek41_contract::own_ops::rope::MAX_POSITIONS) * 32 * 2 * 4,
                ),
            });
        }
        let kv = KvExtent {
            storage: KvStorage::Native,
            pages: 1,
            extent,
            sequences: 4,
            max_pages: 1,
        };
        let metadata = dspark::metadata(
            g,
            &[DraftInput {
                block: DraftBlock {
                    request: 7,
                    slot: 3,
                    main_rows: 3,
                    compute_rows: 5,
                    emit_rows: 2,
                    verify_rows: 3,
                },
                bonus: 4,
            }],
            DraftMode::Greedy,
        )
        .unwrap();
        for stage in 0..g.dspark_stages() {
            let bound = buffers
                .iter()
                .map(|b| dspark::stage_views(b, g, stage).unwrap())
                .collect::<Vec<_>>();
            let mut programs = Vec::new();
            for rank in 0..shape.ranks() as usize {
                let stream = streams[rank];
                let alloc = |bytes| {
                    DeviceBufferSpanV1::new(
                        host.allocate_device_buffer(pools[rank], stream, bytes, 256)
                            .unwrap(),
                        0,
                        bytes,
                    )
                };
                let upload = |span: DeviceBufferSpanV1, bytes: &[u8]| {
                    services
                        .copy_host_to_device(
                            stream,
                            DeviceBufferSpanV1::new(span.buffer, span.offset, bytes.len() as u64),
                            bytes,
                        )
                        .unwrap()
                };
                upload(inputs[rank].live_tokens, &3_u32.to_le_bytes());
                upload(
                    inputs[rank].positions,
                    &[0_i32, 1, 2]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                );
                let row_slots = alloc(12);
                upload(
                    row_slots,
                    &[3_i32; 3]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                );
                let last = alloc(16);
                upload(
                    last,
                    &[-1_i32, -1, -1, 2]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>(),
                );
                let base = Lowering {
                    prefill_row_origin: None,
                    shape: &shape,
                    ops: &ops,
                    buffers: &bound[rank],
                    peers: &bound,
                    inputs: &inputs[rank],
                    rank: rank as u32,
                    kv,
                    head_rows: 3,
                    prefill_hc_fusion: true,
                };
                let k = u64::from(g.shape.hidden) * g.shape.dspark.targets.len() as u64;
                let main =
                    dspark::rings::committed(&shape, extent, &bound[rank].state, stage).unwrap();
                let program = dspark::catch_up(
                    &draft,
                    &base,
                    stage,
                    3,
                    &CatchUpViews {
                        raw_features: alloc(3 * k * 2),
                        feature_codes: alloc(3 * k),
                        feature_scales: alloc(3 * k / 32),
                        ring: main,
                        row_slots,
                        last,
                    },
                )
                .unwrap();
                // Three rows of KV projection take the exact-row decode
                // entry, not the prefill-shaped fold.
                assert!(
                    program
                        .iter()
                        .all(|l| !l.launch.symbol.contains("mxfp8_fold_m8192")),
                    "{}",
                    shape.topology()
                );
                programs.push(program);
            }
            launch(services, &streams, &symbols, programs);
            let mut programs = Vec::new();
            for rank in 0..shape.ranks() as usize {
                let stream = streams[rank];
                let alloc = |bytes| {
                    DeviceBufferSpanV1::new(
                        host.allocate_device_buffer(pools[rank], stream, bytes, 256)
                            .unwrap(),
                        0,
                        bytes,
                    )
                };
                let upload = |span: DeviceBufferSpanV1, bytes: &[u8]| {
                    services
                        .copy_host_to_device(
                            stream,
                            DeviceBufferSpanV1::new(span.buffer, span.offset, bytes.len() as u64),
                            bytes,
                        )
                        .unwrap()
                };
                let words = |values: &[i32]| {
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                };
                upload(inputs[rank].live_tokens, &5_u32.to_le_bytes());
                upload(bound[rank].scratch.bias_select, &metadata.bias_select);
                upload(
                    inputs[rank].token_ids,
                    &metadata
                        .tokens
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                upload(inputs[rank].positions, &words(&metadata.positions));
                upload(inputs[rank].token_batch, &words(&metadata.row_items));
                let first = alloc(4);
                upload(first, &words(&metadata.first));
                let last = alloc(4);
                upload(last, &words(&metadata.last));
                let keys = alloc(metadata.keys.len() as u64 * 4);
                upload(keys, &words(&metadata.keys));
                let item_slots = alloc(metadata.staged_slots.len() as u64 * 4);
                upload(item_slots, &words(&metadata.staged_slots));
                let cells = g.shape.attention.window + 5;
                let width = u64::from(g.shape.attention.head_dim);
                let window = Window {
                    codes: alloc(u64::from(cells) * width),
                    scales: alloc(u64::from(cells) * width / 32),
                    sequences: 1,
                    cells,
                    code_stride: u64::from(cells) * width,
                    scale_stride: u64::from(cells) * width / 32,
                };
                let main =
                    dspark::rings::committed(&shape, extent, &bound[rank].state, stage).unwrap();
                for copy in
                    dspark::rings::stage(main, window, &[3], g.shape.attention.head_dim).unwrap()
                {
                    services
                        .copy_device_to_device(stream, copy.destination, copy.source)
                        .unwrap();
                }
                let phase = Phase::Verify { rows: 5, items: 1 };
                let base = Lowering {
                    prefill_row_origin: None,
                    shape: &shape,
                    ops: &ops,
                    buffers: &bound[rank],
                    peers: &bound,
                    inputs: &inputs[rank],
                    rank: rank as u32,
                    kv,
                    head_rows: 5,
                    prefill_hc_fusion: true,
                };
                let mut program = if stage == 0 {
                    dspark::initialize(&base, phase).unwrap()
                } else {
                    Vec::new()
                };
                program.extend(
                    dspark::block(
                        &draft,
                        &base,
                        stage,
                        phase,
                        BlockViews {
                            window,
                            item_slots,
                            first,
                            last,
                            keys,
                        },
                    )
                    .unwrap(),
                );
                programs.push(program);
            }
            launch(services, &streams, &symbols, programs);
        }
        let journal = host.journal.borrow();
        for part in [
            "lagged_mix_collapse",
            "window_store",
            "latent_sources",
            "moe",
            "expand_f32",
        ] {
            assert!(
                journal
                    .iter()
                    .any(|line| line.starts_with("launch") && line.contains(part)),
                "{} lacks {part}",
                shape.topology()
            );
        }
        eprintln!("DSpark complete block journal: {}", shape.topology());
    }
}
