//! Compare original commit chunks with one contiguous projection launch.
//! This covers GEMM output and status bits; served captures cover the chain.
use super::mxfp8_fold_decode as fixture;
use bundle_deepseek41::sm120::{mxfp8_fold as registration, prefill_matrix, prefill_tma_scales};
use kernel_core::backend::{Backend, Counts, Need, Prepared, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_harness::launch::Bindings;
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold_decode::Entry;
use model_deepseek41_contract::{geometry::FLASH, serving::PREFILL_TOKENS};
use serde_json::json;
use std::collections::BTreeMap;

fn chunks() -> Vec<Vec<u32>> {
    vec![
        vec![33, 31],
        vec![33, 32],
        vec![63, 64],
        vec![64, 64],
        vec![128, 64],
        vec![128, 65],
        vec![128, 127],
        vec![128, 128],
        [128, 64, 128, 65, 128, 127, 128, 128].repeat(2),
        [128, 64, 128, 65, 128, 127, 128, 128].repeat(4),
        vec![128; 64],
    ]
}

fn backend() -> LabResult<Sm120> {
    prefill_tma_scales::register(
        prefill_matrix::register(
            registration::register(Sm120::new().with_packed_fold_weights(), &FLASH, 4)?,
            &FLASH,
            4,
        )?,
        &FLASH,
        4,
    )
}

fn entries() -> Vec<Entry> {
    [(5120, 15360), (512, 5120)]
        .into_iter()
        .map(|(n, k)| {
            registration::decode_entries(&FLASH, 4)
                .unwrap()
                .into_iter()
                .find(|entry| {
                    !entry.paired
                        && entry.query.n == n
                        && entry.query.k == k
                        && entry.query.groups == 1
                })
                .expect("served drafter projection")
        })
        .collect()
}

fn prepare(
    backend: &Sm120,
    entry: &Entry,
    rows: u32,
) -> LabResult<Prepared<Mxfp8FoldGemmBuffers<View>>> {
    let q = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
        m_capacity: PREFILL_TOKENS,
        ..entry.query
    };
    // These cases avoid the one/two-row main stream specialization. Keep
    // this assertion so a later serving selection cannot silently drift.
    assert!(!bundle_deepseek41::sm120::mxfp8_fold_stream::qualified(
        &FLASH, q, rows, false
    ));
    assert!(
        bundle_deepseek41::sm120::mxfp8_fold_direct::selected(&FLASH, q, rows, false).is_none()
    );
    let result = if rows <= 64 {
        backend.mxfp8_fold_gemm(
            &registration::routed_query(q, rows),
            Need::rows(rows).with_exact(kernel_core::properties::Invariance::any_rows(rows)),
        )
    } else if q.k == 5120 && rows >= 128 {
        backend.mxfp8_prefill_gemm(&q, Need::rows(PREFILL_TOKENS))
    } else {
        backend.mxfp8_fold_gemm(&q, Need::rows(PREFILL_TOKENS))
    };
    result.map_err(|e| e.to_string())
}

#[test]
fn commit_projection_boundary_paths_prepare_on_cpu() {
    let backend = backend().unwrap();
    for entry in entries() {
        for chunks in chunks() {
            for rows in chunks.iter().copied().chain([chunks.iter().sum()]) {
                let op = prepare(&backend, &entry, rows).unwrap();
                assert!(!op.info().symbols.is_empty());
            }
        }
    }
}

fn compare(session: &Session, backend: &Sm120, entry: &Entry) -> LabResult<usize> {
    let q = entry.query;
    let weights = fixture::Weights::new(session, entry)?;
    let mut count = 0;
    let mut records = Vec::new();
    for chunks in chunks() {
        let rows: u32 = chunks.iter().sum();
        let all_sizes = entry.legacy().map_err(|e| e.to_string())?.bytes(rows);
        let mut inputs = fixture::pattern(all_sizes.activations, 17);
        let scales = fixture::scales(all_sizes.activation_scales, 29);
        let workspace = chunks
            .iter()
            .copied()
            .chain([rows])
            .map(|n| prepare(backend, entry, n).map(|op| op.info().workspace))
            .collect::<LabResult<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap()
            .max(4);
        let storage = [
            all_sizes.activations,
            all_sizes.activation_scales,
            all_sizes.output,
            4,
            u64::from(rows + 2) * 4,
            workspace,
        ]
        .into_iter()
        .map(|bytes| session.allocate(bytes as usize))
        .collect::<LabResult<Vec<_>>>()?;
        let bindings = Bindings {
            buffers: storage
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .chain([(7, &weights.buffers[0]), (8, &weights.buffers[1])])
                .collect(),
        };
        let view = |handle: u64| View {
            handle,
            bytes: bindings.buffers[&handle].byte_len() as u64,
        };
        let buffers = Mxfp8FoldGemmBuffers {
            activations: view(1),
            activation_scales: view(2),
            weights: view(7),
            weight_scales: view(8),
            output: view(3),
            live_rows: view(4),
            status: view(5),
            status_row_offset: 1,
        };
        for invalid in [false, true] {
            if invalid {
                inputs[(u64::from(rows - 1) * u64::from(q.k) + 17) as usize] = 127;
            }
            let run = |first: u32, n: u32| -> LabResult<(Vec<u8>, Vec<u8>)> {
                let codes = (u64::from(first) * u64::from(q.k)) as usize;
                let bytes = (u64::from(n) * u64::from(q.k)) as usize;
                let mut status = vec![0_u8; storage[4].byte_len()];
                status[..4].copy_from_slice(&0x4000_u32.to_le_bytes());
                status[(n as usize + 1) * 4..].fill(0xa5);
                for (i, image) in [
                    (0, inputs[codes..codes + bytes].to_vec()),
                    (1, scales[codes / 32..(codes + bytes) / 32].to_vec()),
                    (2, vec![0xa5; storage[2].byte_len()]),
                    (3, n.to_le_bytes().to_vec()),
                    (4, status.clone()),
                    (5, vec![0xcd; storage[5].byte_len()]),
                ] {
                    storage[i]
                        .copy_from_bytes(&image, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                let calls = prepare(backend, entry, n)?
                    .launches(&buffers, view(6), Counts::rows(n))
                    .map_err(|e| e.to_string())?;
                let functions = fixture::functions(session, &calls)?;
                fixture::run(session, &session.stream, &calls, &functions, &bindings)?;
                let mut output = vec![0; storage[2].byte_len()];
                let mut actual_status = vec![0; storage[4].byte_len()];
                storage[2]
                    .copy_to_bytes(&mut output, &session.stream)
                    .map_err(|e| e.to_string())?;
                storage[4]
                    .copy_to_bytes(&mut actual_status, &session.stream)
                    .map_err(|e| e.to_string())?;
                let end = n as usize * q.n as usize * 4;
                if output[end..].iter().any(|&byte| byte != 0xa5)
                    || actual_status[..4] != status[..4]
                    || actual_status[(n as usize + 1) * 4..] != status[(n as usize + 1) * 4..]
                {
                    return Err(format!(
                        "commit projection guard changed n={} k={} rows={n}",
                        q.n, q.k
                    ));
                }
                output.truncate(end);
                Ok((output, actual_status[4..(n as usize + 1) * 4].to_vec()))
            };
            let mut expected = (Vec::new(), Vec::new());
            let mut first = 0;
            for &n in &chunks {
                let (output, status) = run(first, n)?;
                expected.0.extend(output);
                expected.1.extend(status);
                first += n;
            }
            let actual = run(0, rows)?;
            for (label, a, b) in [
                ("output", &expected.0, &actual.0),
                ("status", &expected.1, &actual.1),
            ] {
                if let Some(at) = a.iter().zip(b).position(|(a, b)| a != b) {
                    return Err(format!(
                        "commit projection {label} differs n={} k={} rows={rows} invalid={invalid} byte={at}",
                        q.n, q.k
                    ));
                }
                if a.len() != b.len() {
                    return Err("commit projection extent differs".into());
                }
            }
            count += 1;
            eprintln!(
                "COMMIT_PROJECTION_BITWISE n={} k={} rows={rows} invalid={invalid} PASS",
                q.n, q.k
            );
            records.push(json!({"n":q.n,"k":q.k,"rows":rows,"chunks":chunks,"invalid":invalid,"bitwise":true}));
        }
    }
    session.receipt(
        &format!("dspark-commit-n{}-k{}", q.n, q.k),
        BTreeMap::from([("comparisons".into(), json!(records))]),
    )?;
    Ok(count)
}

#[test]
fn commit_projection_gpu_preserves_split_row_bits() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("commit projection pair needs idle GPUs").unwrap();
        return;
    }
    let backend = backend().unwrap();
    let result = kernel_harness::gpu_test_multi(2 << 30, |sessions| {
        for session in sessions {
            let mut count = 0;
            for entry in entries() {
                count += compare(session, &backend, &entry)?;
            }
            if count != 44 {
                return Err(format!("commit projection cases {count}, expected44"));
            }
            eprintln!("COMMIT_PROJECTION_COMPLETE cases={count}");
        }
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "the pair executed");
}
