use kernel_core::backend::{Counts, Need, View};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::engram::HashTables;
use model_deepseek41_contract::own_ops::engram_hash::{EngramHash, EngramHashBuffers};

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}
fn op(items: u32) -> EngramHash {
    let e = FLASH.shape.engram;
    EngramHash {
        vocab: FLASH.shape.vocab,
        ngram: e.max_ngram,
        heads: e.heads,
        layers: e.layers.len() as u32,
        items,
        stride: 192,
    }
}
fn buffers(op: EngramHash, rows: u32) -> EngramHashBuffers<View> {
    EngramHashBuffers {
        tokens: view(1, u64::from(rows) * 4),
        token_batch: view(2, u64::from(rows) * 4),
        q_start: view(3, u64::from(op.items) * 4),
        history: view(4, u64::from(op.items) * u64::from(op.ngram - 1) * 8),
        text: view(5, u64::from(rows)),
        constants: view(6, op.constant_words() * 8),
        ids: view(7, op.output_bytes()),
        status: view(8, u64::from(rows) * 4),
    }
}

#[test]
fn engram_hash_preparation_checks_every_span_alias_and_geometry() {
    use model_deepseek41_contract::own_ops::OwnOps;
    let op = op(2);
    let own = crate::own_ops::Sm120Own::new(&FLASH);
    assert!(
        own.engram_hash(
            &EngramHash {
                vocab: op.vocab - 1,
                ..op
            },
            Need::rows(12)
        )
        .is_err()
    );
    let prepared = own.engram_hash(&op, Need::rows(12)).unwrap();
    let b = buffers(op, 12);
    let call = prepared.launches(&b, view(0, 0), Counts::rows(12)).unwrap();
    assert_eq!(call.len(), 1);
    assert!(call[0].plan.pdl);
    // Each row's (layer, column) tasks run in one CTA of whole warps.
    assert_eq!(call[0].symbol, crate::engram_hash_rows::SYMBOL);
    assert_eq!(call[0].plan.grid, [12, 1, 1]);
    let tasks = op.layers * op.columns() as u32;
    assert_eq!(call[0].plan.block, [tasks.div_ceil(32) * 32, 1, 1]);
    for index in 0..8 {
        let mut bad = b;
        let fields = [
            &mut bad.tokens,
            &mut bad.token_batch,
            &mut bad.q_start,
            &mut bad.history,
            &mut bad.text,
            &mut bad.constants,
            &mut bad.ids,
            &mut bad.status,
        ];
        fields[index].bytes -= 1;
        assert!(
            prepared
                .launches(&bad, view(0, 0), Counts::rows(12))
                .is_err(),
            "short operand {index}"
        );
    }
    for input in 1..=8 {
        let bad = EngramHashBuffers {
            ids: view(input, op.output_bytes()),
            ..b
        };
        if input != 7 {
            assert!(
                prepared
                    .launches(&bad, view(0, 0), Counts::rows(12))
                    .is_err()
            );
        }
    }
    assert!(prepared.launches(&b, view(0, 0), Counts::rows(13)).is_err());
    assert!(crate::engram_hash::prepare(EngramHash { items: 33, ..op }, Need::rows(12)).is_err());
}

fn gpu_functions() -> [(&'static str, &'static [u8], &'static str); 2] {
    let symbol = crate::engram_hash_entries::SYMBOL;
    let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).unwrap();
    assert_eq!(name, crate::engram_hash_entries::CUBIN);
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
fn engram_hash_gpu_functions_resolve_in_embedded_cubins() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("engram-hash-symbols-{}", std::process::id()));
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
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
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

fn replay_images(
    input: model_deepseek41_contract::own_ops::engram_hash::HashInput<'_>,
    status: &[u32],
) -> Vec<Vec<u8>> {
    // The dependency producer rewrites U32 words. The kernel still reads
    // one U8 per live row; only the fixture's backing storage is padded.
    let mut text = input.text.to_vec();
    text.resize(text.len().next_multiple_of(4), 0);
    vec![
        input.tokens.iter().flat_map(|x| x.to_le_bytes()).collect(),
        input
            .token_batch
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        input.q_start.iter().flat_map(|x| x.to_le_bytes()).collect(),
        input.history.iter().flat_map(|x| x.to_le_bytes()).collect(),
        text,
        status.iter().flat_map(|x| x.to_le_bytes()).collect(),
    ]
}

#[test]
fn every_hash_fixture_rewrite_is_word_sized_without_changing_text_bytes() {
    use model_deepseek41_contract::own_ops::engram_hash::HashInput;
    for items in [1_usize, 8, 32] {
        for width in 1..=6 {
            let rows = items * width;
            let text = (0..rows).map(|r| u8::from(r % 3 != 1)).collect::<Vec<_>>();
            let images = replay_images(
                HashInput {
                    tokens: &vec![0; rows],
                    token_batch: &vec![0; rows],
                    q_start: &vec![0; items],
                    history: &vec![-1; items * 3],
                    text: &text,
                },
                &vec![0; rows],
            );
            for (operand, bytes) in images.iter().enumerate() {
                assert!(
                    !bytes.is_empty() && bytes.len().is_multiple_of(4),
                    "items={items} width={width} operand={operand} has {} bytes",
                    bytes.len()
                );
            }
            assert_eq!(&images[4][..rows], text);
            assert!(images[4][rows..].iter().all(|&byte| byte == 0));
        }
    }
}

#[test]
fn engram_hash_gpu_matches_integer_oracle_under_poison_and_graph_replay() {
    kernel_harness::gpu_test(64 << 20, |session| {
        use kernel_core::contract::DType;
        use kernel_harness::launch::{Bindings, launch};
        use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
        use model_deepseek41_contract::own_ops::engram_hash::{HashInput, oracle};
        use std::collections::BTreeMap;
        let [(_, cubin, symbol), (_, probes, producer_symbol)] = gpu_functions();
        let module = session.load_cubin(cubin)?;
        let probes = session.load_cubin(probes)?;
        let function = module.function(symbol).map_err(|e| e.to_string())?;
        let (_, rows_cubin) =
            bundle_deepseek41::sm120::fold_cubin_for_symbol(crate::engram_hash_rows::SYMBOL)
                .ok_or("missing row-parallel Engram hash cubin")?;
        let rows_module = session.load_cubin(rows_cubin)?;
        let rows_function = rows_module
            .function(crate::engram_hash_rows::SYMBOL)
            .map_err(|e| e.to_string())?;
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
        let hash = HashTables::of(&FLASH)?;
        let constants: Vec<_> = hash
            .device_words()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let constant_buffer = upload(&constants)?;
        let live = upload(&1_u32.to_le_bytes())?;
        for items in [1_u32, 8, 32] {
            for width in 1..=6_u32 {
                let op = op(items);
                let rows = items * width;
                let mut tokens = (0..rows)
                    .map(|r| (r * 177 + 21) % op.vocab)
                    .collect::<Vec<_>>();
                let batch = (0..rows).map(|r| (r / width) as i32).collect::<Vec<_>>();
                let starts = (0..items)
                    .map(|item| (item * width) as i32)
                    .collect::<Vec<_>>();
                let history = (0..items * 3)
                    .map(|i| if i % 3 == 0 { -1 } else { i64::from(i * 19) })
                    .collect::<Vec<_>>();
                let text = (0..rows).map(|r| u8::from(r % 7 != 3)).collect::<Vec<_>>();
                // One bad live row tests safe indices; its successors see that
                // raw ID only until a DEAD position blocks their lookback.
                if rows > 3 {
                    tokens[2] = op.vocab;
                }
                let seed = vec![0_u32; rows as usize];
                let mut expected_status = seed.clone();
                let mut expected = vec![u32::MAX; op.output_bytes() as usize / 4];
                oracle(
                    op,
                    &hash,
                    HashInput {
                        tokens: &tokens,
                        token_batch: &batch,
                        q_start: &starts,
                        history: &history,
                        text: &text,
                    },
                    &mut expected,
                    &mut expected_status,
                )
                .map_err(|e| e.to_string())?;
                let prepared = crate::engram_hash::prepare(op, Need::rows(rows))
                    .map_err(|e| format!("{e:?}"))?;
                let call = prepared
                    .launches(&buffers(op, rows), view(0, 0), Counts::rows(rows))
                    .map_err(|e| e.to_string())?
                    .remove(0);
                // The served row-parallel entry and the serial entry it replaces
                // both match the integer oracle under the same replays.
                assert_eq!(call.symbol, crate::engram_hash_rows::SYMBOL);
                let serial = kernel_core::backend::Launch {
                    symbol: crate::engram_hash_entries::SYMBOL.into(),
                    plan: kernel_core::contract::LaunchPlan {
                        grid: [rows.div_ceil(crate::engram_hash_entries::THREADS), 1, 1],
                        block: [crate::engram_hash_entries::THREADS, 1, 1],
                        ..call.plan
                    },
                    ..call.clone()
                };
                for (function, call) in [(&rows_function, &call), (&function, &serial)] {
                    let images = replay_images(
                        HashInput {
                            tokens: &tokens,
                            token_batch: &batch,
                            q_start: &starts,
                            history: &history,
                            text: &text,
                        },
                        &seed,
                    );
                    let inputs = images
                        .iter()
                        .map(|bytes| upload(bytes))
                        .collect::<Result<Vec<_>, _>>()?;
                    let staged = images
                        .iter()
                        .map(|bytes| upload(bytes))
                        .collect::<Result<Vec<_>, _>>()?;
                    let rewrites =
                        Rewrites::new(session, &staged.iter().zip(&inputs).collect::<Vec<_>>())?;
                    let output = session.allocate(op.output_bytes() as usize)?;
                    let bindings = Bindings {
                        buffers: BTreeMap::from([
                            (1, &inputs[0]),
                            (2, &inputs[1]),
                            (3, &inputs[2]),
                            (4, &inputs[3]),
                            (5, &inputs[4]),
                            (6, &constant_buffer),
                            (7, &output),
                            (8, &inputs[5]),
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
                    let expected: Vec<_> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let ranges = (0..op.layers)
                        .map(|layer| {
                            let start = (u64::from(layer) * u64::from(op.stride) * op.columns() * 4)
                                as usize;
                            start..start + (u64::from(rows) * op.columns() * 4) as usize
                        })
                        .collect::<Vec<_>>();
                    let mut replay = Replay::new(session).output_complement(
                        "hash-indices",
                        &output,
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
                    let mut actual = vec![0; expected.len()];
                    output
                        .copy_to_bytes(&mut actual, &session.stream)
                        .map_err(|e| e.to_string())?;
                    if actual != expected {
                        return Err(format!(
                            "{} hash differs at items={items}, width={width}",
                            call.symbol
                        ));
                    }
                    let mut status = vec![0; rows as usize * 4];
                    inputs[5]
                        .copy_to_bytes(&mut status, &session.stream)
                        .map_err(|e| e.to_string())?;
                    let expected_status: Vec<_> = expected_status
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    if status != expected_status {
                        return Err("hash sticky status differs".into());
                    }
                    let mut receipt = BTreeMap::from([
                        ("items".into(), serde_json::json!(items)),
                        ("width".into(), serde_json::json!(width)),
                        ("bitwise_oracle".into(), serde_json::json!(true)),
                        ("entry".into(), serde_json::json!(call.symbol)),
                    ]);
                    replay.record(&mut receipt);
                    session.receipt("deepseek41-engram-device-hash", receipt)?;
                }
            }
        }
        Ok(())
    })
    .expect("device Engram hash matches exact integers under replay and poison");
}
