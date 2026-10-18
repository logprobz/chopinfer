//! Real code/value and scale pools crossing both 32-bit address lines.

#[path = "common/pool_offsets.rs"]
mod offsets;

use kernel_core::attention::BlockTables;
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::compressed_kv::{
    CompressedKvPages, CompressedKvStorage, CompressedKvWriteBuffers,
};
use kernel_core::pooled_indexer::{PositionKeyPages, PositionKeyWriteBuffers};
use kernel_core::precision::quantizer::BlockQuantization;
use kernel_core::quantize::Scales;
use kernel_lab::families::{fp4_key_write, pooling_write_mutants::Fault};
use kernel_lab::kit::cases::fp4_key_write::{Query, queries};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::backend::Sm120;
use runtime_cuda::DeviceBuffer;
use serde_json::json;
use std::collections::BTreeMap;

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn view(handle: u64, buffer: &DeviceBuffer) -> View {
    View {
        handle,
        bytes: buffer.byte_len() as u64,
    }
}

fn geometry(query: Query) -> (usize, usize, bool, BlockQuantization) {
    match query {
        Query::Index(q) => (
            q.head_dim as usize,
            q.page_size as usize,
            true,
            BlockQuantization::E2M1Ceil32,
        ),
        Query::Compressed(q) => (
            q.width as usize,
            q.page_size as usize,
            q.storage == CompressedKvStorage::Packed,
            BlockQuantization::E2M1E4M3Scale16,
        ),
    }
}

/// Both rows straddle a page boundary. All index operands have magnitude
/// one, so key normalization and identity RoPE are exact before A1's oracle.
fn run(session: &Session, query: Query, strides: [u64; 2], fault: Option<Fault>) -> LabResult<()> {
    let (width, page, packed, format) = geometry(query);
    let row_bytes = width * if packed { 1 } else { 4 } / 2;
    let scale_bytes = width / format.group_size() as usize;
    let [code_stride, scale_stride] = strides;
    let mut codes = offsets::SentinelPool::new(session, code_stride + (page * row_bytes) as u64)?;
    let mut scales = if packed {
        Some(offsets::SentinelPool::new(
            session,
            scale_stride + (page * scale_bytes) as u64,
        )?)
    } else {
        None
    };
    let keys: Vec<f32> = (0..2)
        .flat_map(|row| {
            (0..width).map(move |d| {
                if matches!(query, Query::Index(_)) {
                    if (d + row) % 2 == 0 { 1.0 } else { -1.0 }
                } else {
                    [6.0, -1.0, 2.0, -3.0, 4.0, -0.5, 1.0, -2.0][(d + row) % 8]
                }
            })
        })
        .collect();
    let expected: Vec<_> = keys
        .chunks_exact(width)
        .map(|row| {
            let quantized = format.apply(row, 1, width).map_err(|e| e.0)?;
            let scales = match quantized.scales {
                Scales::Ue8M0(v) | Scales::Ue4M3(v) => v,
                _ => unreachable!(),
            };
            let values = if packed {
                quantized.codes
            } else {
                format
                    .fake_quant(row, 1, width)
                    .map_err(|e| e.0)?
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            };
            Ok::<_, String>((values, scales))
        })
        .collect::<Result<_, _>>()?;
    let key = upload(
        session,
        &keys
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let norm = upload(
        session,
        &vec![0x3f80_u16; width]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let table: Vec<u8> = (0..132 * 32)
        .flat_map(|_| [1.0_f32, 0.0].into_iter().flat_map(f32::to_le_bytes))
        .collect();
    let rope = upload(session, &table)?;
    let positions = upload(
        session,
        &[(page - 1) as i32, page as i32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let batch = upload(session, &[0; 8])?;
    let live = upload(session, &2_u32.to_le_bytes())?;
    let status = upload(session, &[0; 20])?;
    let tables = upload(
        session,
        &[0_i32, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let page_tables = BlockTables {
        tables: view(7, &tables),
        sequences: 1,
        max_pages: 2,
    };
    let backend = Sm120::new().with_fp4_key_writes(fp4_key_write::entries());
    let counts = Counts::rows(2).with_positions(132);
    let launches = match query {
        Query::Index(q) => backend
            .position_indexer_key_write(&q, Need::rows(2))
            .map_err(|e| e.to_string())?
            .launches(
                &PositionKeyWriteBuffers {
                    status: view(10, &status),
                    status_row_offset: 3,
                    key: view(1, &key),
                    norm_weight: view(2, &norm),
                    cos_sin: view(3, &rope),
                    positions: view(4, &positions),
                    row_batch: view(5, &batch),
                    live_rows: view(6, &live),
                    keys: PositionKeyPages {
                        codes: view(8, &codes.buffer),
                        scales: view(9, &scales.as_ref().unwrap().buffer),
                        pages: 2,
                        code_page_stride: code_stride,
                        scale_page_stride: scale_stride,
                        page_tables,
                    },
                },
                View::EMPTY,
                counts,
            ),
        Query::Compressed(q) => backend
            .compressed_kv_write(&q, Need::rows(2))
            .map_err(|e| e.to_string())?
            .launches(
                &CompressedKvWriteBuffers {
                    status: view(10, &status),
                    status_row_offset: 3,
                    latent: view(1, &key),
                    cos_sin: view(3, &rope),
                    positions: view(4, &positions),
                    row_batch: view(5, &batch),
                    live_rows: view(6, &live),
                    cache: CompressedKvPages {
                        values: view(8, &codes.buffer),
                        scales: scales.as_ref().map(|s| view(9, &s.buffer)),
                        pages: 2,
                        value_page_stride: code_stride,
                        scale_page_stride: scale_stride,
                        page_tables,
                    },
                },
                View::EMPTY,
                counts,
            ),
    }
    .map_err(|e| e.to_string())?;
    let mut bindings = launch::Bindings {
        buffers: BTreeMap::from([
            (1, &key),
            (2, &norm),
            (3, &rope),
            (4, &positions),
            (5, &batch),
            (6, &live),
            (7, &tables),
            (8, &codes.buffer),
            (10, &status),
        ]),
    };
    if let Some(scales) = &scales {
        bindings.buffers.insert(9, &scales.buffer);
    }
    let module = session
        .module(&fault.map_or_else(|| fp4_key_write::CUBIN.to_owned(), Fault::cubin_name))?;
    for call in launches {
        let symbol = fault.map_or_else(|| call.symbol.clone(), |f| f.symbol(&call.symbol));
        let function = module.function(&symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            call.plan,
            &call.arguments,
            &bindings,
        )?;
    }
    session.stream.synchronize().map_err(|e| e.to_string())?;
    for (row, (expected_codes, expected_scales)) in expected.iter().enumerate() {
        let at = if row == 0 {
            ((page - 1) * row_bytes) as u64
        } else {
            code_stride
        };
        offsets::same(
            "FP4 wide writer",
            "code/value bytes",
            expected_codes,
            &codes.read(session, at, row_bytes as u64)?,
        )?;
        codes.mark(at, row_bytes as u64);
        if let Some(scales) = &mut scales {
            let at = if row == 0 {
                ((page - 1) * scale_bytes) as u64
            } else {
                scale_stride
            };
            offsets::same(
                "FP4 wide writer",
                "scale bytes",
                expected_scales,
                &scales.read(session, at, scale_bytes as u64)?,
            )?;
            scales.mark(at, scale_bytes as u64);
        }
    }
    let mut scanned = codes.check_untouched(session, "FP4 code/value sentinels")?;
    if let Some(scales) = &scales {
        scanned += scales.check_untouched(session, "FP4 scale sentinels")?;
    }
    offsets::receipt(
        session,
        "fp4_key_write",
        &format!("{query:?}"),
        json!({
            "code_stride": code_stride, "scale_stride": scale_stride,
            "oracle": "A1 trained FP4", "sentinel_bytes_checked": scanned,
        }),
    )
}

#[test]
fn fp4_planes_cross_both_32_bit_boundaries() {
    gpu_test(10 << 30, |session| {
        let [signed, unsigned] = offsets::FP4_PLANE_STRIDES;
        let mut checked = 0;
        for query in queries() {
            let (width, page, packed, format) = geometry(query);
            let row_bytes = width * if packed { 1 } else { 4 } / 2;
            // The low and high runs use the same inputs and exact A1 oracle.
            run(
                session,
                query,
                [
                    (page * row_bytes + 4096) as u64,
                    (page * width / format.group_size() as usize + 4096) as u64,
                ],
                None,
            )?;
            for strides in [[signed, unsigned], [unsigned, signed]] {
                run(session, query, strides, None)?;
                checked += 1;
            }
        }
        assert_eq!(checked, 12);
        // Narrowed products stay inside the real allocation at offset 4096.
        // Detection must be an oracle mismatch, not a CUDA address fault.
        for query in queries().into_iter().filter(|q| {
            !matches!(q,
            Query::Compressed(q) if q.storage == CompressedKvStorage::Bf16)
        }) {
            for fault in [Fault::NarrowCodes, Fault::NarrowScales] {
                let error = run(session, query, [unsigned, unsigned], Some(fault))
                    .expect_err("32-bit writer mutant passed the unchanged wide-plane oracle");
                let plane = if fault == Fault::NarrowCodes {
                    "code/value bytes"
                } else {
                    "scale bytes"
                };
                assert!(
                    error.contains(plane) && error.contains("differs from the low-id run"),
                    "{fault:?}: {error}"
                );
            }
        }
        Ok(())
    })
    .unwrap();
}
