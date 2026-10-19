use kernel_core::backend::{Counts, Need, View};
use model_deepseek41_contract::own_ops::ring_gather::{RingGather, RingGatherBuffers};

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

#[test]
fn ring_gather_preparation_checks_spans_and_aliases() {
    let op = RingGather {
        bytes: 64,
        source_stride: 64,
        destination_stride: 80,
        slots: 4,
    };
    let prepared = crate::ring_gather::prepare(op, Need::rows(3)).unwrap();
    assert!(
        crate::ring_gather::prepare(
            RingGather {
                bytes: u32::MAX,
                source_stride: u32::MAX,
                destination_stride: u32::MAX,
                ..op
            },
            Need::rows(1)
        )
        .is_err()
    );
    assert!(crate::ring_gather::prepare(op, Need::rows(u32::MAX)).is_err());
    let buffers = RingGatherBuffers {
        source: view(1, 256),
        source_slots: view(2, 12),
        live_items: view(3, 4),
        destination: view(4, 224),
    };
    let workspace = view(0, 0);
    let launches = prepared
        .launches(&buffers, workspace, Counts::rows(3))
        .unwrap();
    assert_eq!(launches.len(), 1);
    assert!(launches[0].plan.pdl);
    assert_eq!(launches[0].plan.grid, [3, 1, 1]);
    assert!(
        prepared
            .launches(
                &RingGatherBuffers {
                    destination: view(4, 223),
                    ..buffers
                },
                workspace,
                Counts::rows(3)
            )
            .is_err()
    );
    assert!(
        prepared
            .launches(
                &RingGatherBuffers {
                    source: view(1, 255),
                    ..buffers
                },
                workspace,
                Counts::rows(3)
            )
            .is_err()
    );
    assert!(
        prepared
            .launches(
                &RingGatherBuffers {
                    source_slots: view(2, 11),
                    ..buffers
                },
                workspace,
                Counts::rows(3)
            )
            .is_err()
    );
    assert!(
        prepared
            .launches(
                &RingGatherBuffers {
                    destination: view(1, 256),
                    ..buffers
                },
                workspace,
                Counts::rows(3)
            )
            .is_err()
    );
    assert!(
        prepared
            .launches(&buffers, workspace, Counts::rows(4))
            .is_err()
    );
}

fn gpu_functions() -> [(&'static str, &'static [u8], &'static str); 2] {
    let gather = crate::ring_gather_entries::SYMBOL;
    let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(gather).unwrap();
    assert_eq!(name, crate::ring_gather_entries::CUBIN);
    [
        (name, bytes, gather),
        (
            kernels_sm120::probes::CUBIN,
            bundle_deepseek41::sm120::dependency_probes().0,
            kernels_sm120::probes::LIVE,
        ),
    ]
}

#[test]
fn ring_gather_gpu_functions_resolve_in_embedded_cubins() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("dspark-ring-symbols-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (name, bytes, symbol) in gpu_functions() {
        let path = root.join(format!("{name}.cubin"));
        std::fs::write(&path, bytes).unwrap();
        let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("--dump-elf-symbols")
            .arg(&path)
            .output()
            .expect("inspect embedded cubin symbols without opening CUDA");
        std::fs::remove_file(path).unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        let entries: std::collections::BTreeSet<_> = text
            .lines()
            .filter(|line| {
                line.contains("STT_FUNC")
                    && line.contains("STB_GLOBAL")
                    && line.contains("STO_ENTRY")
            })
            .filter_map(|line| line.split_whitespace().last())
            .collect();
        assert!(
            entries.contains(symbol),
            "{name} has no GPU entry {symbol}; found {entries:?}"
        );
        println!(
            "{}",
            serde_json::json!({"cubin":name, "sha256":kernel_build::digest(bytes),
            "function":symbol, "resolved":true})
        );
    }
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn ring_gather_gpu_bytes_poison_and_graph_replay() {
    kernel_harness::gpu_test(64 << 20, |session| {
        use kernel_core::contract::DType;
        use kernel_harness::launch::{Bindings, launch};
        use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
        use model_deepseek41_contract::own_ops::ring_gather::oracle;
        use std::collections::BTreeMap;
        const ITEMS: u32 = 7;
        let [(_, cubin, gather_symbol), (_, probes, producer_symbol)] = gpu_functions();
        let module = session.load_cubin(cubin)?;
        let probes = session.load_cubin(probes)?;
        let producer = LiveProducer::with_function(
            session,
            probes
                .function(producer_symbol)
                .map_err(|e| e.to_string())?,
        )?;
        let upload = |bytes: &[u8]| {
            let buffer = session.allocate(bytes.len())?;
            buffer
                .copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(buffer)
        };
        for op in [
            RingGather {
                bytes: 128 * 512,
                source_stride: 128 * 512,
                destination_stride: 133 * 512,
                slots: 8,
            },
            RingGather {
                bytes: 128 * 16,
                source_stride: 128 * 16,
                destination_stride: 133 * 16,
                slots: 8,
            },
            RingGather {
                bytes: 7,
                source_stride: 11,
                destination_stride: 13,
                slots: 8,
            },
        ] {
            let source: Vec<u8> = (0..op.source_stride as usize * op.slots as usize)
                .map(|i| (i.wrapping_mul(37) ^ (i >> 8)) as u8)
                .collect();
            let slots = [7_u32, 1, 4, u32::MAX, 0, 2, 3];
            let ids: Vec<_> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
            let src = upload(&source)?;
            let src_image = upload(&source)?;
            let indices = upload(&ids)?;
            let indices_image = upload(&ids)?;
            let live = upload(&ITEMS.to_le_bytes())?;
            let length = (ITEMS * op.destination_stride) as usize;
            let destination = session.allocate(length)?;
            let prepared =
                crate::ring_gather::prepare(op, Need::rows(ITEMS)).map_err(|e| format!("{e:?}"))?;
            let buffers = RingGatherBuffers {
                source: view(1, source.len() as u64),
                source_slots: view(2, ids.len() as u64),
                live_items: view(3, 4),
                destination: view(4, length as u64),
            };
            let call = prepared
                .launches(&buffers, view(0, 0), Counts::rows(ITEMS))
                .map_err(|e| e.to_string())?
                .remove(0);
            assert_eq!(call.symbol, gather_symbol);
            let function = module.function(gather_symbol).map_err(|e| e.to_string())?;
            let bindings = Bindings {
                buffers: BTreeMap::from([(1, &src), (2, &indices), (3, &live), (4, &destination)]),
            };
            let invoke = |stream: &runtime_cuda::Stream| {
                launch(
                    &session.context,
                    &function,
                    stream,
                    call.plan,
                    &call.arguments,
                    &bindings,
                )
            };
            let graph = session.capture(invoke)?;
            let rewrites =
                Rewrites::new(session, &[(&src_image, &src), (&indices_image, &indices)])?;
            for count in [0_u32, 1, 3, ITEMS] {
                live.copy_from_bytes(&count.to_le_bytes(), &session.stream)
                    .map_err(|e| e.to_string())?;
                let mut expected = vec![0xff; length];
                oracle(op, &source, &slots, count, &mut expected).map_err(|e| e.to_string())?;
                let ranges: Vec<_> = (0..count as usize)
                    .filter(|&i| slots[i] < op.slots)
                    .map(|i| {
                        i * op.destination_stride as usize
                            ..i * op.destination_stride as usize + op.bytes as usize
                    })
                    .collect();
                let mut replay = Replay::new(session).output_complement(
                    "ring",
                    &destination,
                    WriteMap::of(DType::U8).ranges(ranges)?,
                    expected.clone(),
                );
                replay.eager(|| invoke(&session.stream))?;
                replay.relaunch(|| invoke(&session.stream))?;
                for _ in 0..3 {
                    replay.graph(&graph)?;
                }
                replay.pdl(
                    || producer.launch_rewriting(session, &live, &rewrites),
                    || invoke(&session.stream),
                    call.plan.pdl,
                )?;
                let mut actual = vec![0; length];
                destination
                    .copy_to_bytes(&mut actual, &session.stream)
                    .map_err(|e| e.to_string())?;
                if actual != expected {
                    return Err(format!(
                        "ring gather differs at live={count}, bytes={}",
                        op.bytes
                    ));
                }
                let mut receipt = BTreeMap::from([
                    ("bytes".into(), serde_json::json!(op.bytes)),
                    ("live_items".into(), serde_json::json!(count)),
                    ("bitwise_oracle".into(), serde_json::json!(true)),
                ]);
                replay.record(&mut receipt);
                session.receipt("deepseek41-dspark-ring-gather", receipt)?;
            }
        }
        Ok(())
    })
    .expect("ring gather matches its byte oracle under replay and poison");
}
