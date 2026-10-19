//! Paired old/new measurements on every serving Flash dense shape.
mod modules;
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

fn check(report: &kit::Report, expected: usize, gpu: bool) {
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

fn functions(session: &Session, calls: &[Launch]) -> LabResult<Vec<Function>> {
    calls
        .iter()
        .map(|call| {
            let (_, bytes) =
                fold_cubin_for_symbol(&call.symbol).ok_or("missing serving fold cubin")?;
            session
                .load_cubin(bytes)?
                .function(&call.symbol)
                .map_err(|e| e.to_string())
        })
        .collect()
}

fn run(
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
    let backend = Sm120::new()
        .with_mxfp8_fold(old)
        .with_mxfp8_fold_decode(entry.clone());
    let new_query = if rows <= 32 {
        q
    } else {
        kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
            m_capacity: PREFILL_TOKENS,
            ..q
        }
    };
    let single = if !entry.paired {
        Some(
            backend
                .mxfp8_fold_gemm(&new_query, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let pair = if entry.paired {
        Some(
            backend
                .mxfp8_fold_gemm_pair(&q, Need::rows(rows))
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let workspace = single.as_ref().map_or_else(
        || pair.as_ref().unwrap().info().workspace,
        |op| op.info().workspace,
    );
    let sizes = entry.legacy().map_err(|e| e.to_string())?.bytes(rows);
    const OFFSET: usize = 3;
    let pattern = |bytes: u64, seed: usize| {
        (0..bytes as usize)
            .map(|i| {
                let x = i.wrapping_mul(1664525).wrapping_add(seed * 1013904223);
                ((x >> 13) % 120) as u8 | if (x >> 7) & 1 != 0 { 128 } else { 0 }
            })
            .collect::<Vec<_>>()
    };
    let scales = |bytes: u64, seed: usize| {
        (0..bytes as usize)
            .map(|i| 117 + ((i * 7 + i / 11 + seed) % 17) as u8)
            .collect::<Vec<_>>()
    };
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
        pattern(sizes.weights, 2),
        scales(sizes.weight_scales, 2),
        vec![0xa5; sizes.output as usize],
        rows.to_ne_bytes().to_vec(),
        initial_status.clone(),
        vec![0xcd; workspace.max(4) as usize],
        pattern(if entry.paired { sizes.weights } else { 4 }, 3),
        scales(if entry.paired { sizes.weight_scales } else { 4 }, 3),
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        vec![0xa5; sizes.output as usize],
        initial_status,
    ];
    let buffers = images
        .iter()
        .map(|bytes| session.allocate(bytes.len()))
        .collect::<LabResult<Vec<_>>>()?;
    let view = |i: usize| View {
        handle: i as u64 + 1,
        bytes: images[i].len() as u64,
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
    let new_calls = if let Some(op) = single {
        op.launches(&new_first, view(7), Counts::rows(rows))
    } else {
        pair.unwrap()
            .launches(&[new_first, new_second], view(7), Counts::rows(rows))
    }
    .map_err(|e| e.to_string())?;
    let old_functions = functions(session, &old_calls)?;
    let new_functions = functions(session, &new_calls)?;
    let bindings = Bindings {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64 + 1, b))
            .collect(),
    };
    let upload = |images: &[Vec<u8>]| -> LabResult<()> {
        for (buffer, bytes) in buffers.iter().zip(images) {
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
    // Capture the real three-stage chain once and replay it with changing
    // live counts and poisoned partials. The original path runs eagerly.
    upload(&images)?;
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
    println!(
        "{}",
        json!({"phase":"bitwise-complete","n":q.n,"k":q.k,"groups":q.groups,"rows":rows,"paired":entry.paired,"ranks":ranks,"bitwise":true})
    );
    let old_timing = kernel_harness::time_named(session, "mxfp8-fold-baseline", 2, 7, |stream| {
        run(session, stream, &old_calls, &old_functions, &bindings)
    })?;
    let new_timing = kernel_harness::time_named(session, "mxfp8-fold-decode", 2, 7, |stream| {
        run(session, stream, &new_calls, &new_functions, &bindings)
    })?;
    Ok(
        json!({"n":q.n,"k":q.k,"groups":q.groups,"rows":rows,"paired":entry.paired,"ranks":ranks,
        "bitwise":true,"baseline_us":old_timing.median_ms*1000.0,"decode_us":new_timing.median_ms*1000.0,
        "baseline_launches":old_calls.len(),"decode_launches":new_calls.len()}),
    )
}

#[test]
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
