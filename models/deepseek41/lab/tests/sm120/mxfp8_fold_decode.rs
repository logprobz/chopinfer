//! Paired old/new measurements on every serving Flash dense shape.
mod concurrent;
pub(super) mod modules;
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, mxfp8_fold as registration};
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_harness::kit::{self, OpResult, Outcome, Required, cases::mxfp8_fold as cases};
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_decode::Entry;
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS, geometry::FLASH, serving::PREFILL_TOKENS,
};
use runtime_cuda::{Function, Stream};
use serde_json::json;
use std::collections::BTreeMap;

// time_ab reports the median of B/A pairs from alternating AB and BA rounds.
const MAX_PAIRED_RATIO: f32 = 1.03;

fn accepts_paired_ratio(ratio: f32, unchanged: bool) -> bool {
    ratio.is_finite() && ratio > 0.0 && (unchanged || ratio <= MAX_PAIRED_RATIO)
}

#[test]
fn timing_gate_allows_three_percent_interleaved_median_noise() {
    for ratio in [0.8, 1.0, 1.001_422_5, 1.03] {
        assert!(accepts_paired_ratio(ratio, false), "ratio={ratio}");
    }
    for ratio in [1.0301, 1.04, 0.0, f32::NAN, f32::INFINITY] {
        assert!(!accepts_paired_ratio(ratio, false), "ratio={ratio}");
    }
    assert!(accepts_paired_ratio(1.04, true));
    assert!(!accepts_paired_ratio(f32::NAN, true));
}

fn entries() -> Vec<Entry> {
    registration::decode_catalog(&FLASH).unwrap()
}

fn topologies(entry: &Entry) -> Vec<u32> {
    SUPPORTED_RANKS
        .iter()
        .copied()
        .filter(|&ranks| {
            registration::decode_entries(&FLASH, ranks)
                .unwrap()
                .iter()
                .any(|e| {
                    e.symbol == entry.symbol && e.query == entry.query && e.paired == entry.paired
                })
        })
        .collect()
}

#[test]
fn decode_fold_topologies_are_covered() {
    let tested = entries();
    for &ranks in SUPPORTED_RANKS {
        for enabled in registration::decode_entries(&FLASH, ranks).unwrap() {
            assert!(
                tested.iter().any(|entry| entry.symbol == enabled.symbol
                    && entry.query == enabled.query
                    && entry.paired == enabled.paired),
                "missing TP{ranks} comparison: {}",
                enabled.symbol
            );
            if enabled.paired {
                assert_eq!(enabled.query.n, FLASH.shape.moe.intermediate / ranks);
            }
        }
    }
    let (_, conformance) = required();
    assert_eq!(
        conformance.len(),
        tested.iter().filter(|entry| !entry.paired).count()
    );
}

fn required() -> (Sm120, Vec<Required<'static>>) {
    let mut backend = Sm120::new();
    let mut required = Vec::new();
    for entry in entries().into_iter().filter(|e| !e.paired) {
        let q = entry.query;
        let ranks = topologies(&entry);
        backend = backend.with_mxfp8_fold_decode(entry);
        required.push(
            Required::new(
                format!("decode ranks={ranks:?} {q:?}"),
                cases::need(32),
                move |backend, need| backend.mxfp8_fold_gemm(&q, need),
                move |op| cases::case(q, 32, op),
            )
            .required(),
        );
    }
    (backend, required)
}

pub(super) fn check(report: &kit::Report, expected: usize, gpu: bool) {
    assert!(expected > 0);
    assert_eq!(report.prepared().len(), expected);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required fold refused")
        };
        for name in ["declaration", "need", "conformance", kit::PDL] {
            if !gpu && !["declaration", "need"].contains(&name) {
                continue;
            }
            assert!(
                matches!(
                    checks.iter().find(|c| c.name == name).unwrap().outcome,
                    Outcome::Passed(_)
                ),
                "{} {name}",
                op.name
            );
        }
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn decode_fold_dry_run() {
    println!("{}", modules::check().unwrap());
    let cubins = modules::timing_cubins().unwrap();
    assert!(!cubins.bytes("probe_sm120").unwrap().is_empty());
    let (backend, required) = required();
    let report = kit::conform(&backend, &super::mxfp8_fold::ModelModules, None, &required).unwrap();
    check(&report, required.len(), false);
    for &ranks in SUPPORTED_RANKS {
        let registered = registration::decode_registrations(&FLASH, ranks, 0).unwrap();
        for id in [
            "layer.wq_a",
            "layer.wq_b",
            "layer.wkv",
            "layer.wo_a",
            "layer.wo_b",
            "layer.shared.gate-up",
            "layer.shared.down",
            "csa.indexer.wq_b",
            "engram.wkv",
            "dspark.main_proj",
        ] {
            for rows in registration::DECODE_BUCKETS {
                assert!(
                    registered
                        .iter()
                        .any(|r| r.projection == id && r.usage.capacity == rows)
                );
            }
        }
    }
    for entry in entries() {
        assert!(!topologies(&entry).is_empty());
        for rows in registration::DECODE_BUCKETS {
            let backend = Sm120::new().with_mxfp8_fold_decode(entry.clone());
            let info = if entry.paired {
                backend
                    .mxfp8_fold_gemm_pair(&entry.query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            } else {
                backend
                    .mxfp8_fold_gemm(&entry.query, Need::rows(rows))
                    .unwrap()
                    .info()
                    .clone()
            };
            assert_eq!(info.workspace, entry.workspace(rows).unwrap());
            for (symbol, _) in info.symbols {
                assert!(fold_cubin_for_symbol(&symbol).is_some());
            }
        }
    }
    println!(
        "validated {} distinct Flash TP1/TP2/TP4 shapes at rows 1,2,4,8,16,32 and prefill 64",
        entries().len()
    );
}

pub(super) fn functions(session: &Session, calls: &[Launch]) -> LabResult<Vec<Function>> {
    #[derive(Default)]
    struct Cached(std::cell::RefCell<BTreeMap<String, runtime_cuda::Module>>);
    let cached = session.state(|| Ok(Cached::default()))?;
    calls
        .iter()
        .map(|call| {
            let (name, bytes) = if call.symbol.contains("_mxfp8_stream_") {
                super::mxfp8_fold_stream::cubin()?
            } else {
                fold_cubin_for_symbol(&call.symbol).ok_or("missing serving fold cubin")?
            };
            let mut modules = cached.0.borrow_mut();
            let module = match modules.entry(name.into()) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(session.load_cubin(bytes)?)
                }
            };
            module.function(&call.symbol).map_err(|e| e.to_string())
        })
        .collect()
}

pub(super) fn run(
    session: &Session,
    stream: &Stream,
    calls: &[Launch],
    functions: &[Function],
    bindings: &Bindings<'_>,
) -> LabResult<()> {
    for (call, function) in calls.iter().zip(functions) {
        launch(
            &session.context,
            function,
            stream,
            call.plan,
            &call.arguments,
            bindings,
        )?;
    }
    Ok(())
}

fn compare(session: &Session, entry: &Entry, rows: u32) -> LabResult<serde_json::Value> {
    compare_with(session, entry, rows, None, None, true, false)
}

pub(super) struct Weights {
    buffers: Vec<runtime_cuda::DeviceBuffer>,
}
pub(super) fn pattern(bytes: u64, seed: usize) -> Vec<u8> {
    (0..bytes as usize)
        .map(|i| {
            let x = i.wrapping_mul(1664525).wrapping_add(seed * 1013904223);
            ((x >> 13) % 120) as u8 | if (x >> 7) & 1 != 0 { 128 } else { 0 }
        })
        .collect()
}
pub(super) fn scales(bytes: u64, seed: usize) -> Vec<u8> {
    (0..bytes as usize)
        .map(|i| 117 + ((i * 7 + i / 11 + seed) % 17) as u8)
        .collect()
}
impl Weights {
    pub(super) fn new(session: &Session, entry: &Entry) -> LabResult<Self> {
        let sizes = entry.legacy().map_err(|e| e.to_string())?.bytes(1);
        let images = [
            pattern(sizes.weights, 2),
            scales(sizes.weight_scales, 2),
            pattern(if entry.paired { sizes.weights } else { 4 }, 3),
            scales(if entry.paired { sizes.weight_scales } else { 4 }, 3),
        ];
        let buffers = images
            .iter()
            .map(|bytes| {
                let b = session.allocate(bytes.len())?;
                b.copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
                Ok(b)
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self { buffers })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compare_with(
    session: &Session,
    entry: &Entry,
    rows: u32,
    ticket: Option<&kernels_sm120::gemm::mxfp8_fold_ticket::Entry>,
    weights: Option<&Weights>,
    timed: bool,
    profile: bool,
) -> LabResult<serde_json::Value> {
    compare_impl(session, entry, rows, ticket, weights, timed, profile, None)
}

pub(super) fn compare_stream(
    session: &Session,
    entry: &Entry,
    rows: u32,
    candidate: &kernels_sm120::gemm::mxfp8_fold_stream::Entry,
    weights: &Weights,
    profile: bool,
) -> LabResult<serde_json::Value> {
    compare_impl(
        session,
        entry,
        rows,
        Some(&candidate.ticket),
        Some(weights),
        true,
        profile,
        Some(candidate),
    )
}

pub(super) fn compare_stream_deferred(
    session: &Session,
    entry: &Entry,
    rows: u32,
    candidate: &kernels_sm120::gemm::mxfp8_fold_stream::Entry,
    weights: &Weights,
) -> LabResult<serde_json::Value> {
    compare_impl(
        session,
        entry,
        rows,
        Some(&candidate.ticket),
        Some(weights),
        false,
        false,
        Some(candidate),
    )
}

#[allow(clippy::too_many_arguments)]
fn compare_impl(
    session: &Session,
    entry: &Entry,
    rows: u32,
    ticket: Option<&kernels_sm120::gemm::mxfp8_fold_ticket::Entry>,
    weights: Option<&Weights>,
    timed: bool,
    profile: bool,
    stream_entry: Option<&kernels_sm120::gemm::mxfp8_fold_stream::Entry>,
) -> LabResult<serde_json::Value> {
    let q = entry.query;
    let ranks = topologies(entry);
    let old = registration::entries(
        &FLASH,
        *ranks.first().ok_or("entry has no enabled topology")?,
    )?
    .into_iter()
    .find(|old| {
        old.query
            == kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: PREFILL_TOKENS,
                ..q
            }
    })
    .ok_or("missing prefill-shaped baseline")?;
    let old_backend = Sm120::new().with_mxfp8_fold(old.clone());
    let old_op = old_backend
        .mxfp8_fold_gemm(&old.query, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let mut backend = Sm120::new()
        .with_mxfp8_fold(old)
        .with_mxfp8_fold_decode(entry.clone());
    if let Some(ticket) = ticket {
        backend = backend.with_mxfp8_fold_ticket(ticket.clone());
    }
    let prefetch = ticket.is_some_and(|entry| entry.prefetch_weights);
    let mut baseline_backend = Sm120::new().with_mxfp8_fold_decode(entry.clone());
    let baseline_query = if stream_entry.is_some() {
        baseline_backend = registration::register(Sm120::new(), &FLASH, ranks[0])?;
        registration::routed_query(q, rows)
    } else if prefetch {
        let previous = registration::ticket_catalog(&FLASH)?
            .into_iter()
            .find(|old| old.paired == entry.paired && old.query == ticket.unwrap().query)
            .ok_or("missing accepted ticket baseline")?;
        let query = registration::small_query(&FLASH, previous.query, rows);
        baseline_backend = baseline_backend.with_mxfp8_fold_ticket(previous);
        query
    } else {
        q
    };
    let use_split_baseline = ticket.is_some() && (stream_entry.is_some() || prefetch || rows <= 32);
    let baseline_single = if use_split_baseline && !entry.paired {
        Some(
            baseline_backend
                .mxfp8_fold_gemm(&baseline_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let baseline_pair = if use_split_baseline && entry.paired {
        Some(
            baseline_backend
                .mxfp8_fold_gemm_pair(&baseline_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let baseline_workspace = baseline_single
        .as_ref()
        .map(|op| op.info().workspace)
        .or_else(|| baseline_pair.as_ref().map(|op| op.info().workspace))
        .unwrap_or(0);
    let new_query = if let Some(ticket) = ticket {
        registration::small_query(&FLASH, ticket.query, rows)
    } else if rows <= 32 {
        q
    } else {
        kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
            m_capacity: PREFILL_TOKENS,
            ..q
        }
    };
    let stream_op = stream_entry
        .map(|candidate| {
            Sm120::new()
                .with_mxfp8_fold_stream(candidate.clone())
                .mxfp8_fold_gemm_stream(&candidate.ticket.query, entry.paired, Need::rows(rows))
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let single = if !entry.paired && stream_op.is_none() {
        Some(
            backend
                .mxfp8_fold_gemm(&new_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let pair = if entry.paired && stream_op.is_none() {
        Some(
            backend
                .mxfp8_fold_gemm_pair(&new_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let workspace = stream_op
        .as_ref()
        .map(|op| op.info().workspace)
        .or_else(|| single.as_ref().map(|op| op.info().workspace))
        .or_else(|| pair.as_ref().map(|op| op.info().workspace))
        .unwrap();
    let sizes = entry.legacy().map_err(|e| e.to_string())?.bytes(rows);
    const OFFSET: usize = 3;
    let initial_status: Vec<u8> = (0..rows as usize + OFFSET + 2)
        .flat_map(|i| {
            (if i < OFFSET || i >= OFFSET + rows as usize {
                0x4000_u32
            } else {
                0
            })
            .to_ne_bytes()
        })
        .collect();
    let mut images = vec![
        pattern(sizes.activations, 1),
        scales(sizes.activation_scales, 1),
        if weights.is_some() {
            vec![0; 4]
        } else {
            pattern(sizes.weights, 2)
        },
        if weights.is_some() {
            vec![0; 4]
        } else {
            scales(sizes.weight_scales, 2)
        },
        vec![0xa5; sizes.output as usize],
        rows.to_ne_bytes().to_vec(),
        initial_status.clone(),
        vec![0xcd; workspace.max(4) as usize],
        if weights.is_some() {
            vec![0; 4]
        } else {
            pattern(if entry.paired { sizes.weights } else { 4 }, 3)
        },
        if weights.is_some() {
            vec![0; 4]
        } else {
            scales(if entry.paired { sizes.weight_scales } else { 4 }, 3)
        },
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        initial_status,
        vec![0xcd; baseline_workspace.max(4) as usize],
    ];
    if let Some(candidate) = stream_entry {
        images.extend([
            vec![0xcd; candidate.packed_bytes() as usize],
            vec![
                0xcd;
                if entry.paired {
                    candidate.packed_bytes() as usize
                } else {
                    4
                }
            ],
            vec![0xcd; candidate.state_bytes(rows).map_err(|e| e.to_string())? as usize],
        ]);
    }
    let buffers = images
        .iter()
        .map(|bytes| session.allocate(bytes.len()))
        .collect::<LabResult<Vec<_>>>()?;
    let view = |i: usize| View {
        handle: i as u64 + 1,
        bytes: weights
            .and_then(|w| {
                [2, 3, 8, 9]
                    .iter()
                    .position(|&index| index == i)
                    .map(|index| w.buffers[index].byte_len() as u64)
            })
            .unwrap_or(images[i].len() as u64),
    };
    let first = Mxfp8FoldGemmBuffers {
        activations: view(0),
        activation_scales: view(1),
        weights: view(2),
        weight_scales: view(3),
        output: view(4),
        live_rows: view(5),
        status: view(6),
        status_row_offset: OFFSET as u64,
    };
    let second = Mxfp8FoldGemmBuffers {
        weights: view(8),
        weight_scales: view(9),
        output: view(10),
        ..first
    };
    let new_first = Mxfp8FoldGemmBuffers {
        output: view(11),
        status: view(13),
        ..first
    };
    let new_second = Mxfp8FoldGemmBuffers {
        output: view(12),
        status: view(13),
        ..second
    };
    let mut old_calls = old_op
        .launches(&first, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    if entry.paired {
        old_calls.extend(
            old_op
                .launches(&second, View::EMPTY, Counts::rows(rows))
                .map_err(|e| e.to_string())?,
        );
    }
    if let Some(op) = baseline_single {
        old_calls = op
            .launches(&first, view(14), Counts::rows(rows))
            .map_err(|e| e.to_string())?;
    }
    if let Some(op) = baseline_pair {
        old_calls = op
            .launches(&[first, second], view(14), Counts::rows(rows))
            .map_err(|e| e.to_string())?;
    }
    let stateful = stream_entry.map(|_| kernel_core::mxfp8_gemm::Mxfp8FoldStreamBuffers {
        first: new_first,
        second: entry.paired.then_some(new_second),
        packed_weights: [view(15), if entry.paired { view(16) } else { View::EMPTY }],
        state: view(17),
    });
    let initialization = stream_op
        .as_ref()
        .map(|op| op.initialize(&stateful.unwrap()))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let new_calls = if let Some(op) = stream_op.as_ref() {
        op.launches(&stateful.unwrap(), view(7), Counts::rows(rows))
    } else if let Some(op) = single {
        op.launches(&new_first, view(7), Counts::rows(rows))
    } else {
        pair.unwrap()
            .launches(&[new_first, new_second], view(7), Counts::rows(rows))
    }
    .map_err(|e| e.to_string())?;
    let old_functions = functions(session, &old_calls)?;
    let new_functions = functions(session, &new_calls)?;
    let initialize_functions = functions(session, &initialization)?;
    let mut bindings = Bindings {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    if let Some(weights) = weights {
        for (i, b) in [2, 3, 8, 9].into_iter().zip(&weights.buffers) {
            bindings.buffers.insert(i + 1, b);
        }
    }
    let upload = |images: &[Vec<u8>]| -> LabResult<()> {
        for (index, (buffer, bytes)) in buffers.iter().zip(images).enumerate() {
            if (stream_entry.is_some() && index >= 15)
                || (weights.is_some() && [2, 3, 8, 9].contains(&index))
            {
                continue;
            }
            buffer
                .copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    };
    let read = |index: usize| -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; buffers[index].byte_len()];
        buffers[index]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    // Capture the candidate chain once and replay it with changing live
    // counts and poisoned partials. The baseline runs eagerly here.
    upload(&images)?;
    if stream_entry.is_some() {
        for index in 15..18 {
            buffers[index]
                .copy_from_bytes(&images[index], &session.stream)
                .map_err(|e| e.to_string())?;
        }
        run(
            session,
            &session.stream,
            &initialization,
            &initialize_functions,
            &bindings,
        )?;
    }
    session.stream.synchronize().map_err(|e| e.to_string())?;
    session.stream.begin_capture().map_err(|e| e.to_string())?;
    run(
        session,
        &session.stream,
        &new_calls,
        &new_functions,
        &bindings,
    )?;
    let graph = session
        .stream
        .end_capture()
        .map_err(|e| e.to_string())?
        .instantiate()
        .map_err(|e| e.to_string())?;
    for live in [0, 1, rows.saturating_sub(1), rows] {
        images[5] = live.to_ne_bytes().to_vec();
        upload(&images)?;
        run(
            session,
            &session.stream,
            &old_calls,
            &old_functions,
            &bindings,
        )?;
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
        if ticket.is_some() {
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            run(
                session,
                &session.stream,
                &new_calls,
                &new_functions,
                &bindings,
            )?;
        }
        assert_eq!(read(4)?, read(11)?, "{q:?} rows={rows} live={live}");
        assert_eq!(read(6)?, read(13)?, "status {q:?}");
        if entry.paired {
            assert_eq!(read(10)?, read(12)?, "paired {q:?}");
        }
    }
    // Invalid codes/scales in the last group zero the whole row; sticky
    // bits and status guards survive. These are exact old/new comparisons.
    let at = ((rows - 1) * q.activation_row_stride
        + (q.groups - 1) * q.activation_group_stride
        + q.k / 2) as usize;
    let saved = (images[0][at], images[1][at / 32]);
    for (code, scale, previous) in [
        (127, 127, 0),
        (56, 255, 0),
        (126, 254, 0),
        (1, 0, 0),
        (56, 127, 0x80),
        (126, 127, 0),
        (127, 3, 0),
        (255, 246, 0),
        (1, 2, 0),
    ] {
        images[0][at] = code;
        images[1][at / 32] = scale;
        let status_at = (OFFSET + rows as usize - 1) * 4;
        for i in [6, 13] {
            images[i][status_at..status_at + 4].copy_from_slice(&u32::to_ne_bytes(previous));
        }
        upload(&images)?;
        run(
            session,
            &session.stream,
            &old_calls,
            &old_functions,
            &bindings,
        )?;
        run(
            session,
            &session.stream,
            &new_calls,
            &new_functions,
            &bindings,
        )?;
        assert_eq!(
            read(4)?,
            read(11)?,
            "admission {q:?} code={code} scale={scale}"
        );
        assert_eq!(read(6)?, read(13)?);
        if entry.paired {
            assert_eq!(read(10)?, read(12)?);
        }
    }
    images[0][at] = saved.0;
    images[1][at / 32] = saved.1;
    let status_at = (OFFSET + rows as usize - 1) * 4;
    for i in [6, 13] {
        images[i][status_at..status_at + 4].fill(0);
    }
    upload(&images)?;
    if (prefetch || stream_entry.is_some()) && profile {
        concurrent::check(
            session,
            concurrent::Case {
                images: &images,
                buffers: &buffers,
                bindings: &bindings,
                calls: &new_calls,
                functions: &new_functions,
                baseline_calls: &old_calls,
                baseline_functions: &old_functions,
                paired: entry.paired,
                persistent: stream_entry
                    .map(|_| (&initialization[..], &initialize_functions[..], 17)),
            },
        )?;
        upload(&images)?;
    }
    if stream_entry.is_some() {
        assert!(
            read(17)?.iter().all(|&v| v == 0),
            "persistent fold state did not self-reset"
        );
    }
    println!(
        "{}",
        json!({"phase":"bitwise-complete","n":q.n,"k":q.k,"groups":q.groups,"rows":rows,"paired":entry.paired,"ranks":ranks,"bitwise":true})
    );
    let mut record = json!({"n":q.n,"k":q.k,"groups":q.groups,"rows":rows,"paired":entry.paired,"ranks":ranks,
        "bitwise":true,"baseline_launches":old_calls.len(),"decode_launches":new_calls.len()});
    if (prefetch || stream_entry.is_some()) && profile {
        record["concurrent_replays"] = json!(3);
    }
    if stream_entry.is_some() {
        record["baseline_symbols"] = json!(old_calls.iter().map(|c| &c.symbol).collect::<Vec<_>>());
        record["candidate_symbols"] =
            json!(new_calls.iter().map(|c| &c.symbol).collect::<Vec<_>>());
    }
    if !timed {
        return Ok(record);
    }
    if let Some(ticket) = ticket {
        let old_graph = session
            .capture(|stream| run(session, stream, &old_calls, &old_functions, &bindings))?;
        let mode = kernel_harness::eviction::EvictionMode::selected()?;
        let paired = kernel_harness::with_timing_label("mxfp8-fold-ticket-ab", || {
            kernel_harness::time_ab(session, 2, 5, [mode, mode], |arm, stream| {
                if arm == 0 {
                    old_graph.launch(stream)
                } else {
                    graph.launch(stream)
                }
                .map_err(|e| e.to_string())
            })
        })?;
        let weight_bytes = (sizes.weights + sizes.weight_scales) * if entry.paired { 2 } else { 1 };
        let floor_us = weight_bytes as f64 / 1_600_000.0;
        record["baseline_us"] = json!(paired.arms[0].median_ms * 1000.0);
        record["decode_us"] = json!(paired.arms[1].median_ms * 1000.0);
        record["paired_ratio"] = json!(paired.paired_ratio);
        record["weight_bytes"] = json!(weight_bytes);
        record["floor_us"] = json!(floor_us);
        record["floor_ratio"] = json!(f64::from(paired.arms[1].median_ms) * 1000.0 / floor_us);
        let unchanged = stream_entry.is_none() && new_query.m_capacity == 32;
        if unchanged {
            assert_eq!(old_calls.len(), new_calls.len());
            for (old, new) in old_calls.iter().zip(&new_calls) {
                assert_eq!(old.symbol, new.symbol);
                assert_eq!(old.plan, new.plan);
            }
        }
        // An identical launch sequence is a retained baseline, not a new
        // implementation subject to a noisy self-comparison timing gate.
        record["same_implementation"] = json!(unchanged);
        record["routing"] = json!(if stream_entry.is_some() {
            "stream"
        } else if unchanged {
            "split-fold"
        } else if prefetch {
            "prefetch"
        } else {
            "ticket"
        });
        record["baseline_routing"] = json!(if stream_entry.is_some() {
            match registration::decode_path(&FLASH, q, rows, entry.paired) {
                registration::DecodePath::Split => "split-fold",
                registration::DecodePath::Ticket => "ticket",
                registration::DecodePath::Prefetch => "prefetch",
            }
        } else if prefetch && !unchanged {
            "ticket"
        } else if rows <= 32 {
            "split-fold"
        } else {
            "prefill"
        });
        record["no_regression"] = json!(accepts_paired_ratio(paired.paired_ratio, unchanged));
        record["regression_limit_ratio"] = json!(MAX_PAIRED_RATIO);
        record["timing_statistic"] = json!("median of interleaved candidate/baseline ratios");
        record["tile_n"] = json!(if unchanged { 16 } else { ticket.tile_n() });
        record["folds_per_cta"] = json!(if unchanged {
            2
        } else {
            ticket.folds_per_cta(rows)
        });
        record["workspace_bytes"] = json!(workspace);
        if stream_entry.is_none() && profile {
            let stage_times = |calls: &[Launch], functions: &[Function]| -> LabResult<Vec<f32>> {
                (0..calls.len())
                    .map(|i| {
                        kernel_harness::time_prepared(
                            session,
                            1,
                            3,
                            mode,
                            |stream| run(session, stream, &calls[..i], &functions[..i], &bindings),
                            |stream| {
                                run(
                                    session,
                                    stream,
                                    &calls[i..i + 1],
                                    &functions[i..i + 1],
                                    &bindings,
                                )
                            },
                        )
                        .map(|t| t.median_ms * 1000.0)
                    })
                    .collect()
            };
            record["baseline_stage_us"] = json!(stage_times(&old_calls, &old_functions)?);
            record["ticket_stage_us"] = json!(stage_times(&new_calls, &new_functions)?);
            record["baseline_grid"] =
                json!(old_calls.iter().map(|c| c.plan.grid).collect::<Vec<_>>());
            record["ticket_grid"] =
                json!(new_calls.iter().map(|c| c.plan.grid).collect::<Vec<_>>());
        }
    } else {
        let old = kernel_harness::time_named(session, "mxfp8-fold-baseline", 2, 7, |stream| {
            run(session, stream, &old_calls, &old_functions, &bindings)
        })?;
        let new = kernel_harness::time_named(session, "mxfp8-fold-decode", 2, 7, |stream| {
            run(session, stream, &new_calls, &new_functions, &bindings)
        })?;
        record["baseline_us"] = json!(old.median_ms * 1000.0);
        record["decode_us"] = json!(new.median_ms * 1000.0);
    }
    Ok(record)
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_FOLD_CUBINS"]
fn decode_fold_gpu() {
    let cubins = modules::timing_cubins().unwrap();
    kernel_harness::gpu_test_multi_with(cubins, 3 << 30, |sessions| {
        let (backend, required) = required();
        let report = kit::conform(
            &backend,
            &super::mxfp8_fold::ModelModules,
            Some(&sessions[0]),
            &required,
        )?;
        check(&report, required.len(), true);
        println!(
            "{}",
            json!({"phase":"conformance-complete","cases":required.len(),"ranks":SUPPORTED_RANKS})
        );
        for session in sessions {
            let mut records = Vec::new();
            for entry in entries() {
                for rows in registration::DECODE_BUCKETS {
                    records.push(compare(session, &entry, rows)?);
                }
                if !entry.paired {
                    records.push(compare(session, &entry, 64)?);
                }
            }
            println!("{}", json!({"comparisons":records}));
            session.receipt(
                "mxfp8-fold-decode",
                BTreeMap::from([
                    ("comparisons".into(), json!(records)),
                    ("conformance".into(), report.json()),
                    ("power_cap_watts".into(), json!(250)),
                ]),
            )?;
        }
        Ok(())
    })
    .unwrap();
}
