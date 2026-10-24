//! The verify allocation kernel against the SDK rule (the contract's
//! oracle), bit for bit, under poisoned relaunch, graph replay and a PDL
//! producer that rewrites every input.
use kernel_core::backend::{Counts, Need, View};
use model_deepseek41_contract::own_ops::allocation::*;

fn views(op: VerifyAllocate) -> VerifyAllocateBuffers<View> {
    let v = |handle, bytes| View { handle, bytes };
    let items = u64::from(op.items);
    VerifyAllocateBuffers {
        items: v(1, items * ITEM_WORDS as u64 * 4),
        scores: v(2, items * BLOCK as u64 * 4),
        rule: v(3, RULE_WORDS as u64 * 4),
        live_items: v(4, 4),
        status: v(5, 4),
        verified: v(6, items * 4),
        starts: v(7, items * 4),
        rows: v(8, u64::from(op.width) * 4),
        expand: v(9, u64::from(op.reserved_rows) * 4),
    }
}

const NO_VIEW: View = View {
    handle: 0,
    bytes: 0,
};

fn gpu_functions() -> [(&'static str, &'static [u8], &'static str); 2] {
    let symbol = crate::allocate_entries::SYMBOL;
    let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).unwrap();
    assert_eq!(name, crate::allocate_entries::CUBIN);
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
fn allocate_preparation_checks_spans_aliases_and_capacity() {
    use model_deepseek41_contract::own_ops::OwnOps;
    let op = VerifyAllocate {
        items: 4,
        reserved_rows: 24,
        width: 12,
    };
    assert!(
        crate::own_ops::Sm120Own::new(&model_deepseek41_contract::geometry::FLASH)
            .dspark_allocate(&op, Need::rows(4))
            .is_ok()
    );
    for bad in [
        VerifyAllocate { items: 0, ..op },
        VerifyAllocate { items: 33, ..op },
        VerifyAllocate {
            reserved_rows: 25,
            ..op
        },
        VerifyAllocate { width: 3, ..op },
        VerifyAllocate { width: 25, ..op },
    ] {
        assert!(
            crate::allocate::prepare(bad, Need::rows(4)).is_err(),
            "{bad:?}"
        );
    }
    let prepared = crate::allocate::prepare(op, Need::rows(4)).unwrap();
    let b = views(op);
    let calls = prepared.launches(&b, NO_VIEW, Counts::rows(4)).unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].plan.pdl);
    assert_eq!(calls[0].plan.grid, [1, 1, 1]);
    // The launch is the same for every live count, so one graph serves all.
    assert_eq!(
        format!("{:?}", calls[0].arguments),
        format!(
            "{:?}",
            prepared.launches(&b, NO_VIEW, Counts::rows(1)).unwrap()[0].arguments
        )
    );
    for index in 0..9 {
        let mut short = b;
        let fields = [
            &mut short.items,
            &mut short.scores,
            &mut short.rule,
            &mut short.live_items,
            &mut short.status,
            &mut short.verified,
            &mut short.starts,
            &mut short.rows,
            &mut short.expand,
        ];
        fields[index].bytes -= 1;
        assert!(prepared.launches(&short, NO_VIEW, Counts::rows(4)).is_err());
    }
    for handle in 1..=4 {
        let aliased = VerifyAllocateBuffers {
            rows: View { handle, bytes: 64 },
            ..b
        };
        assert!(
            prepared
                .launches(&aliased, NO_VIEW, Counts::rows(4))
                .is_err()
        );
    }
    for items in [0, 5] {
        assert!(prepared.launches(&b, NO_VIEW, Counts::rows(items)).is_err());
    }
}

#[test]
fn allocate_gpu_function_resolves_in_the_embedded_cubin() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("allocate-symbols-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let (name, bytes, symbol) = gpu_functions()[0];
    let path = root.join(format!("{name}.cubin"));
    std::fs::write(&path, bytes).unwrap();
    let output = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("--dump-elf-symbols")
        .arg(&path)
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(root).unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(
        text.lines().any(
            |line| line.contains("STO_ENTRY") && line.split_whitespace().last() == Some(symbol)
        ),
        "missing {symbol} in {name}"
    );
}

/// A small deterministic generator, so a failing case reproduces.
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// Items, scores and rule of one case over `count` items. Some cases use
/// coarse scores (equal survivals), NaN and infinite scores, every floor
/// and every width from the items to their reserved rows.
fn case(rng: &mut Rng, count: u32) -> ([u32; RULE_WORDS], Vec<Item>, Vec<[f32; BLOCK]>) {
    let (mut items, mut scores) = (Vec::new(), Vec::new());
    let (mut first_row, mut drafting) = (0, 0);
    let coarse = rng.below(3) == 0;
    for _ in 0..count {
        let reserved = rng.below(BLOCK as u64 + 1) as u32;
        let kind = if rng.below(2) == 0 {
            SamplingKindV1::Greedy
        } else {
            SamplingKindV1::Sampled
        };
        items.push(Item {
            reserved,
            kind,
            score_row: if reserved == 0 { NONE } else { drafting },
            first_row,
        });
        if reserved > 0 {
            drafting += 1;
            scores.push(std::array::from_fn(|_| match rng.below(50) {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                _ if coarse => rng.below(3) as f32 - 1.0,
                _ => (rng.below(2001) as f32 - 1000.0) / 250.0,
            }));
        }
        first_row += 1 + reserved;
    }
    let mut rule = [0; RULE_WORDS];
    rule[0] = count + rng.below(u64::from(first_row - count) + 1) as u32;
    rule[1] = rng.below(BLOCK as u64 + 2) as u32;
    for word in &mut rule[2..] {
        *word = ((rng.below(801) as f32 - 400.0) / 100.0).to_bits();
    }
    (rule, items, scores)
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

#[test]
fn allocate_gpu_matches_the_sdk_rule_under_poison_graph_and_dependency_replay() {
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
            buffer
                .copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(buffer)
        };
        let live = upload(&1_u32.to_le_bytes())?;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut cases = 0;
        for count in [1_u32, 2, 8, 16, 32] {
            for index in 0..24 {
                let (rule, mut items, scores) = case(&mut rng, count);
                // The last cases of each count break an item, which only the
                // status may report.
                let broken = index == 23;
                if broken {
                    items.last_mut().unwrap().first_row += 1;
                }
                let op = VerifyAllocate {
                    items: 32,
                    reserved_rows: 192,
                    width: 192,
                };
                let expected = if broken {
                    None
                } else {
                    Some(allocate_reference(&rule, &items, &scores).map_err(|e| e.to_string())?)
                };
                let mut score_rows = scores.clone();
                score_rows.resize(op.items as usize, [0.0; BLOCK]);
                let images = [
                    {
                        let mut padded = words(items.iter().flat_map(|item| item.encode()));
                        padded.resize(op.items as usize * ITEM_WORDS * 4, 0);
                        padded
                    },
                    score_rows
                        .iter()
                        .flatten()
                        .flat_map(|s| s.to_le_bytes())
                        .collect(),
                    words(rule),
                    words([count]),
                ];
                let inputs = images
                    .iter()
                    .map(|bytes| upload(bytes))
                    .collect::<Result<Vec<_>, _>>()?;
                let staged = images
                    .iter()
                    .map(|bytes| upload(bytes))
                    .collect::<Result<Vec<_>, _>>()?;
                let pairs = staged.iter().zip(&inputs).collect::<Vec<_>>();
                let rewrites = Rewrites::new(session, &pairs)?;
                let reserved_rows: u32 = items.iter().map(|item| 1 + item.reserved).sum();
                let written: [Vec<u8>; 5] = match &expected {
                    Some(allocation) => [
                        words([OK]),
                        words(allocation.verified.iter().copied()),
                        words(allocation.starts.iter().copied()),
                        words(allocation.rows.iter().copied()),
                        words(allocation.expand.iter().copied()),
                    ],
                    None => [words([BAD_INPUT]), vec![], vec![], vec![], vec![]],
                };
                let capacities = [
                    4,
                    op.items as usize * 4,
                    op.items as usize * 4,
                    op.width as usize * 4,
                    op.reserved_rows as usize * 4,
                ];
                let outputs = capacities
                    .iter()
                    .map(|&bytes| session.allocate(bytes))
                    .collect::<Result<Vec<_>, _>>()?;
                let prepared =
                    crate::allocate::prepare(op, Need::rows(32)).map_err(|e| format!("{e:?}"))?;
                let call = prepared
                    .launches(&views(op), NO_VIEW, Counts::rows(32))
                    .map_err(|e| e.to_string())?
                    .remove(0);
                let mut buffers = BTreeMap::new();
                for (handle, buffer) in (1..).zip(inputs.iter().chain(&outputs)) {
                    buffers.insert(handle, buffer);
                }
                let bindings = Bindings { buffers };
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
                let mut replay = Replay::new(session);
                for (k, ((output, bytes), capacity)) in
                    outputs.iter().zip(&written).zip(capacities).enumerate()
                {
                    let mut image = bytes.clone();
                    image.resize(capacity, 0);
                    replay = replay.output_complement(
                        ["status", "verified", "starts", "rows", "expand"][k],
                        output,
                        WriteMap::of(DType::U8).prefix(bytes.len())?,
                        image,
                    );
                }
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
                for (output, bytes) in outputs.iter().zip(&written) {
                    let mut actual = vec![0; bytes.len()];
                    output
                        .copy_to_bytes(&mut actual, &session.stream)
                        .map_err(|e| e.to_string())?;
                    if &actual != bytes {
                        return Err(format!(
                            "allocation differs at count={count} case={index}: {rule:?} {items:?}"
                        ));
                    }
                }
                let mut receipt = BTreeMap::from([
                    ("items".into(), serde_json::json!(count)),
                    ("case".into(), serde_json::json!(index)),
                    ("width".into(), serde_json::json!(rule[0])),
                    ("reserved_rows".into(), serde_json::json!(reserved_rows)),
                    ("broken".into(), serde_json::json!(broken)),
                    ("bitwise_oracle".into(), serde_json::json!(true)),
                ]);
                replay.record(&mut receipt);
                session.receipt("deepseek41-verify-allocation", receipt)?;
                cases += 1;
            }
        }
        assert_eq!(cases, 120);
        Ok(())
    })
    .expect("device allocation matches the SDK rule");
}
