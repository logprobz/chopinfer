//! The allocated acceptance kernel against its contract's oracle, bit for
//! bit, under poisoned relaunch, graph replay and a PDL producer that
//! rewrites every input.
use kernel_core::backend::{Counts, Need, View};
use kernel_core::sampling::OUTPUT_BYTES;
use model_deepseek41_contract::own_ops::acceptance::allocated::*;
use model_deepseek41_contract::own_ops::acceptance::{Acceptance, BLOCK, Input, Item, NONE};

fn views(op: Acceptance, items: u32) -> AllocatedAcceptanceBuffers<View> {
    let v = |handle, bytes| View { handle, bytes };
    AllocatedAcceptanceBuffers {
        items: v(1, u64::from(items) * 40),
        verified: v(2, u64::from(items) * 4),
        status: v(3, 4),
        proposals: v(4, u64::from(op.draft_items) * 20),
        samples: v(5, u64::from(op.sample_rows) * OUTPUT_BYTES as u64),
        stops: v(6, u64::from(op.stop_ids) * 4),
        results: v(7, u64::from(items) * 32),
    }
}

fn gpu_functions() -> [(&'static str, &'static [u8], &'static str); 2] {
    let symbol = crate::allocated_acceptance_entries::SYMBOL;
    let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).unwrap();
    assert_eq!(name, crate::allocated_acceptance_entries::CUBIN);
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
fn allocated_acceptance_preparation_checks_spans_aliases_and_capacity() {
    use model_deepseek41_contract::own_ops::OwnOps;
    let op = Acceptance {
        vocab: 100,
        sample_rows: 24,
        draft_items: 4,
        stop_ids: 4,
    };
    assert!(
        crate::own_ops::Sm120Own::new(&model_deepseek41_contract::geometry::FLASH)
            .dspark_allocated_acceptance(&op, Need::rows(4))
            .is_ok()
    );
    let prepared = crate::allocated_acceptance::prepare(op, Need::rows(4)).unwrap();
    let no_view = View {
        handle: 0,
        bytes: 0,
    };
    let launch = prepared
        .launches(&views(op, 4), no_view, Counts::rows(3))
        .unwrap()
        .remove(0);
    assert_eq!(launch.symbol, crate::allocated_acceptance_entries::SYMBOL);
    assert!(launch.plan.pdl);
    assert_eq!(launch.arguments.0.len(), 12);
    for rows in [0, 5] {
        assert!(
            prepared
                .launches(&views(op, 4), no_view, Counts::rows(rows))
                .is_err()
        );
    }
    let mut short = views(op, 4);
    short.verified.bytes = 8;
    assert!(prepared.launches(&short, no_view, Counts::rows(3)).is_err());
    let mut aliased = views(op, 4);
    aliased.results.handle = aliased.verified.handle;
    assert!(
        prepared
            .launches(&aliased, no_view, Counts::rows(3))
            .is_err()
    );
}

#[test]
fn allocated_acceptance_gpu_function_resolves_in_its_embedded_cubin() {
    let [(name, bytes, symbol), _] = gpu_functions();
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "allocated-acceptance-symbols-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).unwrap();
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
        text.lines().any(|line| line.contains("STT_FUNC")
            && line.contains("STB_GLOBAL")
            && line.contains("STO_ENTRY")
            && line.split_whitespace().last() == Some(symbol)),
        "missing {symbol} in {name}"
    );
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// One allocated step: its operation, descriptors, verified drafts,
/// allocation status, proposals, sample rows and stop ids. `case` selects a
/// failure: 0-3 none, 4 a failed allocation, 5 a verified count above its
/// reservation, 6 a corrupt unverified row, 7 an empty verified row.
#[allow(clippy::type_complexity)]
fn step(
    rng: &mut Rng,
    count: u32,
    case: u32,
) -> (
    Acceptance,
    Vec<Item>,
    Vec<u32>,
    u32,
    Vec<[u32; BLOCK]>,
    Vec<u8>,
    Vec<u32>,
) {
    let (mut items, mut verified, mut proposals) = (Vec::new(), Vec::new(), Vec::new());
    let (mut samples, mut stops) = (Vec::new(), Vec::new());
    for index in 0..count {
        let k = rng.below(BLOCK as u64 + 1) as u32;
        let first_sample = (samples.len() / OUTPUT_BYTES) as u32;
        let draft_item = if k == 0 {
            NONE
        } else {
            proposals.push(std::array::from_fn(|_| 10 + rng.below(3) as u32));
            proposals.len() as u32 - 1
        };
        stops.push(if rng.below(3) == 0 {
            10 + rng.below(3) as u32
        } else {
            90
        });
        items.push(Item {
            epoch: 7 + u64::from(index),
            remaining: 1 + rng.below(8),
            first_sample,
            draft_item,
            k,
            stop_offset: index,
            stop_count: 1,
            reserved: 0,
        });
        let v = rng.below(u64::from(k) + 1) as u32;
        verified.push(v);
        for r in 0..=k {
            let mut bytes = [0xa5; OUTPUT_BYTES];
            let token = if case == 6 && r > v {
                100
            } else {
                10 + rng.below(3) as u32
            };
            let status = u32::from(case == 7 && r == v) * 2;
            bytes[..4].copy_from_slice(&token.to_le_bytes());
            bytes[4..8].copy_from_slice(&status.to_le_bytes());
            bytes[8..16].copy_from_slice(&(-0.25_f64).to_le_bytes());
            samples.extend(bytes);
        }
    }
    if case == 5
        && let Some((index, item)) = items.iter().enumerate().find(|(_, item)| item.k < 5)
    {
        verified[index] = item.k + 1;
    }
    let op = Acceptance {
        vocab: 100,
        sample_rows: (samples.len() / OUTPUT_BYTES) as u32,
        draft_items: proposals.len() as u32,
        stop_ids: count,
    };
    let status = u32::from(case == 4);
    (op, items, verified, status, proposals, samples, stops)
}

#[test]
fn allocated_acceptance_gpu_matches_its_oracle_under_poison_graph_and_dependency_replay() {
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
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut executed = 0;
        for count in [1_u32, 8, 32] {
            for case in 0..8_u32 {
                let (op, items, verified, status, proposals, samples, stops) =
                    step(&mut rng, count, case);
                let expected = oracle(
                    op,
                    Input {
                        items: &items,
                        proposals: &proposals,
                        samples: &samples,
                        stops: &stops,
                    },
                    &verified,
                    status,
                )
                .map_err(|e| e.to_string())?;
                let expected: Vec<_> = expected.iter().flat_map(|d| d.encode()).collect();
                let images = [
                    items.iter().flat_map(|d| d.encode()).collect::<Vec<_>>(),
                    verified.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    status.to_le_bytes().to_vec(),
                    proposals
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
                // A step without drafts has no proposal payload to rewrite.
                let pairs = staged
                    .iter()
                    .zip(&inputs)
                    .zip(&images)
                    .filter(|(_, bytes)| !bytes.is_empty())
                    .map(|(pair, _)| pair)
                    .collect::<Vec<_>>();
                let rewrites = Rewrites::new(session, &pairs)?;
                let output = session.allocate(expected.len() + 32)?;
                let prepared = crate::allocated_acceptance::prepare(op, Need::rows(count))
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
                        (5, &inputs[4]),
                        (6, &inputs[5]),
                        (7, &output),
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
                    "allocated-acceptance",
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
                        "allocated acceptance differs at count={count} case={case}"
                    ));
                }
                let mut receipt = BTreeMap::from([
                    ("items".into(), serde_json::json!(count)),
                    ("case".into(), serde_json::json!(case)),
                    ("bitwise_oracle".into(), serde_json::json!(true)),
                ]);
                replay.record(&mut receipt);
                session.receipt("deepseek41-allocated-acceptance", receipt)?;
                executed += 1;
            }
        }
        if executed != 24 {
            return Err(format!("{executed} of 24 cases executed"));
        }
        Ok(())
    })
    .expect("allocated acceptance matches its oracle");
}

/// The generated cases reach every failure and both stops the walk takes.
#[test]
fn allocated_acceptance_cases_reach_every_outcome() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut failures = std::collections::BTreeSet::new();
    let (mut short, mut rejected) = (0, 0);
    for count in [1_u32, 8, 32] {
        for case in 0..8_u32 {
            let (op, items, verified, status, proposals, samples, stops) =
                step(&mut rng, count, case);
            let decisions = oracle(
                op,
                Input {
                    items: &items,
                    proposals: &proposals,
                    samples: &samples,
                    stops: &stops,
                },
                &verified,
                status,
            )
            .unwrap();
            for ((decision, item), &v) in decisions.iter().zip(&items).zip(&verified) {
                failures.insert(decision.failure);
                short += usize::from(decision.failure == OK && v < item.k);
                rejected += usize::from(decision.flags & 2 != 0);
            }
        }
    }
    assert_eq!(
        failures,
        [OK, BAD_DESCRIPTOR, EMPTY, BAD_SAMPLE]
            .into_iter()
            .collect()
    );
    assert!(short > 10 && rejected > 10, "{short} {rejected}");
}
