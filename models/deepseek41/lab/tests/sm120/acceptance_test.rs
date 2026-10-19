use kernel_core::backend::{Counts, Need, View};
use kernel_core::sampling::OUTPUT_BYTES;
use model_deepseek41_contract::own_ops::acceptance::*;
fn views(op: Acceptance, items: u32) -> AcceptanceBuffers<View> {
    let v = |handle, bytes| View { handle, bytes };
    AcceptanceBuffers {
        items: v(1, u64::from(items) * 40),
        proposals: v(2, u64::from(op.draft_items) * 20),
        samples: v(3, u64::from(op.sample_rows) * OUTPUT_BYTES as u64),
        stops: v(4, u64::from(op.stop_ids) * 4),
        results: v(5, u64::from(items) * 32),
    }
}
fn gpu_functions() -> [(&'static str, &'static [u8], &'static str); 2] {
    let symbol = crate::acceptance_entries::SYMBOL;
    let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).unwrap();
    assert_eq!(name, crate::acceptance_entries::CUBIN);
    [
        (name, bytes, symbol),
        (
            kernels_sm120::probes::CUBIN,
            bundle_deepseek41::sm120::dependency_probes().0,
            kernels_sm120::probes::LIVE,
        ),
    ]
}
#[test]
fn acceptance_gpu_functions_resolve_in_embedded_cubins() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("acceptance-symbols-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (name, bytes, symbol) in gpu_functions() {
        let path = root.join(format!("{name}.cubin"));
        std::fs::write(&path, bytes).unwrap();
        let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("--dump-elf-symbols")
            .arg(&path)
            .output()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(
            text.lines().any(|line| line.contains("STT_FUNC")
                && line.contains("STB_GLOBAL")
                && line.contains("STO_ENTRY")
                && line.split_whitespace().last() == Some(symbol)),
            "missing {symbol} in {name}"
        );
        println!(
            "{}",
            serde_json::json!({"cubin":name,"function":symbol,"resolved":true,"sha256":kernel_build::digest(bytes)})
        );
    }
    std::fs::remove_dir(root).unwrap();
}
#[test]
fn acceptance_preparation_checks_spans_aliases_capacity_and_binding() {
    use model_deepseek41_contract::own_ops::OwnOps;
    let op = Acceptance {
        vocab: 100,
        sample_rows: 12,
        draft_items: 2,
        stop_ids: 2,
    };
    assert!(
        crate::own_ops::Sm120Own::new(&model_deepseek41_contract::geometry::FLASH)
            .dspark_acceptance(&op, Need::rows(2))
            .is_err()
    );
    let prepared = crate::acceptance::prepare(op, Need::rows(2)).unwrap();
    let b = views(op, 2);
    let calls = prepared
        .launches(
            &b,
            View {
                handle: 0,
                bytes: 0,
            },
            Counts::rows(2),
        )
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].plan.pdl);
    for index in 0..5 {
        let mut short = b;
        let fields = [
            &mut short.items,
            &mut short.proposals,
            &mut short.samples,
            &mut short.stops,
            &mut short.results,
        ];
        fields[index].bytes -= 1;
        assert!(
            prepared
                .launches(
                    &short,
                    View {
                        handle: 0,
                        bytes: 0
                    },
                    Counts::rows(2)
                )
                .is_err()
        );
    }
    for handle in 1..=4 {
        let bad = AcceptanceBuffers {
            results: View { handle, bytes: 64 },
            ..b
        };
        assert!(
            prepared
                .launches(
                    &bad,
                    View {
                        handle: 0,
                        bytes: 0
                    },
                    Counts::rows(2)
                )
                .is_err()
        );
    }
    for items in [0, 3] {
        assert!(
            prepared
                .launches(
                    &b,
                    View {
                        handle: 0,
                        bytes: 0
                    },
                    Counts::rows(items)
                )
                .is_err()
        );
    }
}

#[test]
fn acceptance_gpu_matches_host_walk_under_poison_graph_and_dependency_replay() {
    kernel_harness::gpu_test(64 << 20, |session| {
        use kernel_core::contract::DType;
        use kernel_harness::launch::{Bindings, launch};
        use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
        use std::collections::BTreeMap;
        let [(_, cubin, symbol), (_, probes, producer_symbol)] = gpu_functions();
        let module = session.load_cubin(cubin)?;
        let probes = session.load_cubin(probes)?;
        let function = module.function(symbol).map_err(|e| e.to_string())?;
        let producer = LiveProducer::with_function(
            session,
            probes
                .function(producer_symbol)
                .map_err(|e| e.to_string())?,
        )?;
        let upload = |bytes: &[u8]| {
            let buffer = session.allocate(bytes.len().max(4))?;
            if !bytes.is_empty() {
                buffer
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            Ok::<_, String>(buffer)
        };
        let live = upload(&1_u32.to_le_bytes())?;
        for count in [1_u32, 8, 32] {
            for k in 0..=5_u32 {
                for case in 0..8_u32 {
                    let rows = count * (k + 1);
                    let op = Acceptance {
                        vocab: 100,
                        sample_rows: (rows + 1).min(192),
                        draft_items: if k == 0 { 0 } else { count },
                        stop_ids: count,
                    };
                    let mut descriptors = Vec::new();
                    let proposed = vec![[10, 11, 12, 13, 14]; op.draft_items as usize];
                    let mut samples = vec![0xa5; op.sample_rows as usize * OUTPUT_BYTES];
                    let mut stops = Vec::new();
                    for item in 0..count {
                        let depth = item % (k + 1);
                        descriptors.push(Item {
                            epoch: 7 + u64::from(item),
                            remaining: if case == 2 {
                                1 + u64::from(item % 6)
                            } else {
                                9
                            },
                            first_sample: item * (k + 1),
                            draft_item: if k == 0 { NONE } else { item },
                            k,
                            stop_offset: item,
                            stop_count: 1,
                            reserved: 0,
                        });
                        stops.push(if case == 1 { 10 } else { 90 });
                        for row in 0..=k {
                            let at = (item * (k + 1) + row) as usize * OUTPUT_BYTES;
                            let token = if (case == 0 && row == depth && depth < k)
                                || (matches!(case, 3 | 7) && row == 0 && k > 0)
                                || (case == 4 && item % 2 == 0 && row == 0 && k > 0)
                            {
                                99_u32
                            } else {
                                10 + row
                            };
                            samples[at..at + 4].copy_from_slice(&token.to_le_bytes());
                            samples[at + 4..at + 8].fill(0);
                            samples[at + 8..at + 16].copy_from_slice(&(-0.25_f64).to_le_bytes());
                            if (case == 3 || case == 4) && row == k {
                                samples[at + 4..at + 8].copy_from_slice(
                                    &(if case == 3 { 3_u32 } else { 2 }).to_le_bytes(),
                                );
                            }
                        }
                    }
                    if case == 5 {
                        descriptors.last_mut().unwrap().reserved = 1;
                    }
                    if case == 6 {
                        descriptors.last_mut().unwrap().first_sample = u32::MAX;
                    }
                    if case == 7 {
                        let at = (rows as usize - 1) * OUTPUT_BYTES + 8;
                        samples[at..at + 8].copy_from_slice(&f64::NAN.to_le_bytes());
                    }
                    let expected = oracle(
                        op,
                        Input {
                            items: &descriptors,
                            proposals: &proposed,
                            samples: &samples,
                            stops: &stops,
                        },
                    )
                    .map_err(|e| e.to_string())?;
                    let expected: Vec<_> = expected.into_iter().flat_map(|d| d.encode()).collect();
                    let images = [
                        descriptors
                            .iter()
                            .flat_map(|d| d.encode())
                            .collect::<Vec<_>>(),
                        proposed
                            .iter()
                            .flatten()
                            .flat_map(|id| id.to_le_bytes())
                            .collect(),
                        samples,
                        stops.iter().flat_map(|id| id.to_le_bytes()).collect(),
                    ];
                    let inputs = images
                        .iter()
                        .map(|bytes| upload(bytes))
                        .collect::<Result<Vec<_>, _>>()?;
                    let staged = images
                        .iter()
                        .map(|bytes| upload(bytes))
                        .collect::<Result<Vec<_>, _>>()?;
                    // Zero-length proposal inputs at K=0 have no dependency payload.
                    let pairs = staged
                        .iter()
                        .zip(&inputs)
                        .zip(&images)
                        .filter(|(_, bytes)| !bytes.is_empty())
                        .map(|(pair, _)| pair)
                        .collect::<Vec<_>>();
                    let rewrites = Rewrites::new(session, &pairs)?;
                    let output = session.allocate(expected.len() + 32)?;
                    let prepared = crate::acceptance::prepare(op, Need::rows(count))
                        .map_err(|e| format!("{e:?}"))?;
                    let call = prepared
                        .launches(
                            &views(op, count),
                            View {
                                handle: 0,
                                bytes: 0,
                            },
                            Counts::rows(count),
                        )
                        .map_err(|e| e.to_string())?
                        .remove(0);
                    let bindings = Bindings {
                        buffers: BTreeMap::from([
                            (1, &inputs[0]),
                            (2, &inputs[1]),
                            (3, &inputs[2]),
                            (4, &inputs[3]),
                            (5, &output),
                        ]),
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
                    let mut poison_image = expected.clone();
                    poison_image.resize(expected.len() + 32, 0);
                    let mut replay = Replay::new(session).output_complement(
                        "acceptance-counts",
                        &output,
                        WriteMap::of(DType::U8).prefix(expected.len())?,
                        poison_image,
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
                    let mut actual = vec![0; expected.len()];
                    output
                        .copy_to_bytes(&mut actual, &session.stream)
                        .map_err(|e| e.to_string())?;
                    if actual != expected {
                        return Err(format!(
                            "acceptance differs at count={count} k={k} case={case}"
                        ));
                    }
                    let mut receipt = BTreeMap::from([
                        ("items".into(), serde_json::json!(count)),
                        ("k".into(), serde_json::json!(k)),
                        ("case".into(), serde_json::json!(case)),
                        ("bitwise_oracle".into(), serde_json::json!(true)),
                    ]);
                    replay.record(&mut receipt);
                    session.receipt("deepseek41-device-acceptance", receipt)?;
                }
            }
        }
        Ok(())
    })
    .expect("device acceptance preserves the host walk");
}
