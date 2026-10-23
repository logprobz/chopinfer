//! Exact bytes, inactive writes, guarded tails and changing-list graph replay.
//! Build this family with `build_family`; set CHOPINFER_SCALE_EXPAND_CUBINS
//! to its external output directory when running on an idle GPU.

use kernel_core::backend::View;
use kernel_core::contract::Argument;
use kernel_core::precision::scale_offset::{self as codec, decode_bank_matrix};
use kernel_harness::{LabResult, Session, launch, poison};
use kernel_lab_families::moe_scale_expand as family;
use kernels_sm120::moe::scale_expand::{self as expand, ScaleExpandQuery};
use runtime_cuda::{DeviceBuffer, KernelArgument, Module, Stream};
use std::path::PathBuf;

const BANK_GUARD: usize = 16;
const OUTPUT_GUARD: usize = 17;
const TAIL_GUARD: usize = 37;

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Constant,
    Random,
    EveryByte,
    TileEdges,
    Wrapping,
}

struct Fixture {
    query: ScaleExpandQuery,
    bank: Vec<u8>,
    raw: Vec<u8>,
}

fn assemble(blocks: &[Vec<u8>]) -> Vec<u8> {
    let layout =
        codec::bank_layout(&blocks.iter().map(|b| b.len() as u64).collect::<Vec<_>>()).unwrap();
    let mut bank = layout.table;
    for block in blocks {
        bank.extend_from_slice(block);
    }
    assert_eq!(bank.len() as u64, layout.bytes);
    bank
}

fn fixture(query: ScaleExpandQuery, pattern: Pattern) -> Fixture {
    let (rows, groups) = (query.rows as usize, query.groups as usize);
    let matrix_bytes = rows * groups;
    let mut state = 0xd6e8_feb8_6659_fd93_u64;
    let mut blocks = Vec::new();
    let mut raw = Vec::new();
    for matrix in 0..query.experts * query.projections {
        let bytes: Vec<u8> = (0..matrix_bytes)
            .map(|i| match pattern {
                Pattern::Constant => (matrix * 31) as u8,
                Pattern::Random => {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                }
                Pattern::EveryByte => (i + matrix as usize * 37) as u8,
                Pattern::TileEdges => {
                    if i % (64 * groups) <= 1 || i % (64 * groups) == 64 * groups - 1 {
                        253 + (i % 3) as u8
                    } else {
                        120
                    }
                }
                Pattern::Wrapping => 0,
            })
            .collect();
        let block = if matches!(pattern, Pattern::Wrapping) {
            // Valid decoder input beyond what the encoder chooses: bases
            // near 255, offsets that wrap, and exceptions at tile edges.
            let fixed = codec::fixed_bytes(rows as u64, groups as u64, query.width).unwrap();
            let row_bytes = codec::plane_row_bytes(groups as u64, query.width) as usize;
            let mut block = vec![0; fixed as usize];
            block[..rows].fill(255);
            block[rows..rows + rows * row_bytes].fill(255);
            let mut indices = vec![0, matrix_bytes - 1];
            if rows > 64 {
                indices.extend([64 * groups - 1, 64 * groups, 64 * groups + 1]);
            }
            indices.sort_unstable();
            indices.dedup();
            for (n, index) in indices.into_iter().enumerate() {
                block.extend_from_slice(
                    &(((index as u32) << 8) | (253 + n as u32 % 3)).to_le_bytes(),
                );
            }
            block
        } else {
            codec::encode(&bytes, rows as u64, groups as u64, query.width).unwrap()
        };
        if matches!(pattern, Pattern::Wrapping) {
            raw.extend(codec::decode(&block, rows as u64, groups as u64, query.width).unwrap());
        } else {
            raw.extend(bytes);
        }
        blocks.push(block);
    }
    Fixture {
        query,
        bank: assemble(&blocks),
        raw,
    }
}

fn cases() -> Vec<(ScaleExpandQuery, Pattern)> {
    [1, 2, 4]
        .into_iter()
        .flat_map(|width| {
            [
                (1, 1, 1, 1, Pattern::Constant),
                (7, 3, 3, 5, Pattern::Random),
                (63, 17, 2, 5, Pattern::Random),
                (64, 256, 1, 3, Pattern::EveryByte),
                (65, 18, 2, 7, Pattern::TileEdges),
                (129, 33, 2, 4, Pattern::Wrapping),
                (576, 160, 2, 5, Pattern::Random),
                (5120, 18, 1, 5, Pattern::Random),
            ]
            .into_iter()
            .map(move |(rows, groups, projections, experts, p)| {
                (
                    ScaleExpandQuery {
                        width,
                        rows,
                        groups,
                        projections,
                        experts,
                    },
                    p,
                )
            })
        })
        .collect()
}

fn active_lists(experts: u32) -> Vec<Option<Vec<u32>>> {
    let sparse = if experts == 1 {
        vec![0]
    } else {
        vec![experts - 1, 0]
    };
    vec![
        Some((0..experts).rev().collect()),
        Some(sparse),
        Some(vec![]),
        None,
    ]
}

fn active_bytes(experts: u32, ids: &[u32]) -> Vec<u8> {
    let mut words = vec![-1_i32; experts as usize + 1];
    words[0] = ids.len() as i32;
    for (word, id) in words[1..].iter_mut().zip(ids) {
        *word = *id as i32;
    }
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn images(f: &Fixture, ids: Option<&[u32]>, fill: u8) -> (Vec<u8>, Vec<u8>) {
    let q = f.query;
    let mut expected = vec![fill; OUTPUT_GUARD + f.raw.len() + TAIL_GUARD];
    let mut initial = expected.clone();
    let matrix_bytes = (q.rows * q.groups) as usize;
    for expert in 0..q.experts {
        if ids.is_some_and(|ids| !ids.contains(&expert)) {
            continue;
        }
        for projection in 0..q.projections {
            let matrix = (expert * q.projections + projection) as usize;
            let decoded =
                decode_bank_matrix(&f.bank, matrix, q.rows.into(), q.groups.into(), q.width)
                    .unwrap();
            let start = OUTPUT_GUARD + matrix * matrix_bytes;
            expected[start..start + matrix_bytes].copy_from_slice(&decoded);
            // A missing write cannot match even if the scale equals fill.
            for (dst, byte) in initial[start..start + matrix_bytes].iter_mut().zip(decoded) {
                *dst = !byte;
            }
        }
    }
    (initial, expected)
}

#[test]
fn cpu_scale_expand_fixtures_cover_codec_and_write_maps() {
    assert_eq!(cases().len(), 24);
    for (q, pattern) in cases() {
        let f = fixture(q, pattern);
        let matrix_bytes = (q.rows * q.groups) as usize;
        for matrix in 0..(q.experts * q.projections) as usize {
            let decoded =
                decode_bank_matrix(&f.bank, matrix, q.rows.into(), q.groups.into(), q.width)
                    .unwrap();
            assert_eq!(
                decoded,
                f.raw[matrix * matrix_bytes..(matrix + 1) * matrix_bytes]
            );
        }
        if matches!(pattern, Pattern::EveryByte) {
            assert!((0..=255).all(|b| f.raw.contains(&b)));
            let fixed = codec::fixed_bytes(q.rows.into(), q.groups.into(), q.width).unwrap();
            let first = u32::from_le_bytes(f.bank[..4].try_into().unwrap()) as u64;
            let next = u32::from_le_bytes(f.bank[4..8].try_into().unwrap()) as u64;
            assert!((next - first - fixed) / 4 > u64::from(q.rows * q.groups / 2));
        }
        for ids in active_lists(q.experts) {
            let (initial, expected) = images(&f, ids.as_deref(), 0xa5);
            let expert_bytes = matrix_bytes * q.projections as usize;
            for expert in 0..q.experts as usize {
                let start = OUTPUT_GUARD + expert * expert_bytes;
                let writes = ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&(expert as u32)));
                assert!(
                    initial[start..start + expert_bytes]
                        .iter()
                        .zip(&expected[start..start + expert_bytes])
                        .all(|(a, b)| (*a != *b) == writes)
                );
            }
            assert_eq!(&initial[..OUTPUT_GUARD], &expected[..OUTPUT_GUARD]);
            assert_eq!(
                &initial[initial.len() - TAIL_GUARD..],
                &expected[expected.len() - TAIL_GUARD..]
            );
        }
    }
}

fn load(session: &Session) -> LabResult<Module> {
    let directory = std::env::var_os("CHOPINFER_SCALE_EXPAND_CUBINS")
        .map(PathBuf::from)
        .ok_or("CHOPINFER_SCALE_EXPAND_CUBINS is required")?;
    kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
    let bytes = std::fs::read(directory.join(format!("{}.cubin", family::CUBIN)))
        .map_err(|e| e.to_string())?;
    session.load_cubin(&bytes)
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn run_case(session: &Session, module: &Module, f: &Fixture) -> LabResult<()> {
    let q = f.query;
    let mut bank_image = vec![0x6d; BANK_GUARD];
    bank_image.extend_from_slice(&f.bank);
    bank_image.resize(bank_image.len() + TAIL_GUARD, 0x6d);
    let bank = upload(session, &bank_image)?;
    let active = upload(session, &active_bytes(q.experts, &[]))?;
    let output = session.allocate(OUTPUT_GUARD + f.raw.len() + TAIL_GUARD)?;
    let function = module
        .function(expand::symbol(q.width).unwrap())
        .map_err(|e| e.to_string())?;
    let config = launch::config(expand::plan(q).unwrap())?;
    let invoke = |stream: &Stream, list: bool| -> LabResult<()> {
        let args = expand::arguments(
            q,
            View {
                handle: 1,
                bytes: f.bank.len() as u64,
            },
            list.then_some(View {
                handle: 2,
                bytes: active.byte_len() as u64,
            }),
            View {
                handle: 3,
                bytes: f.raw.len() as u64,
            },
        )
        .map_err(|e| e.to_string())?;
        let args = args
            .0
            .iter()
            .map(|a| match a {
                Argument::DeviceBuffer(1) => KernelArgument::Range {
                    buffer: &bank,
                    offset: BANK_GUARD,
                    byte_len: f.bank.len(),
                },
                Argument::DeviceBuffer(2) => KernelArgument::Buffer(&active),
                Argument::DeviceBuffer(3) => KernelArgument::Range {
                    buffer: &output,
                    offset: OUTPUT_GUARD,
                    byte_len: f.raw.len(),
                },
                Argument::U32(v) => KernelArgument::U32(*v),
                Argument::NullDevicePointer => KernelArgument::NullBuffer,
                _ => unreachable!("scale expansion ABI"),
            })
            .collect::<Vec<_>>();
        launch::launch_with(&function, stream, config, &args)
    };
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let listed_graph = session.capture(|stream| invoke(stream, true))?;
    let all_graph = session.capture(|stream| invoke(stream, false))?;
    // Replay the same graph as the device count and expert ordering change.
    for ids in active_lists(q.experts) {
        let active_image = active_bytes(q.experts, ids.as_deref().unwrap_or(&[]));
        poison::write(session, &active, 0, &active_image)?;
        for replay in 0..3 {
            let (initial, expected) =
                images(f, ids.as_deref(), if replay == 1 { 0x5a } else { 0xa5 });
            poison::write(session, &output, 0, &initial)?;
            if replay == 0 {
                invoke(&session.stream, ids.is_some())?;
            } else {
                let graph = if ids.is_some() {
                    &listed_graph
                } else {
                    &all_graph
                };
                graph.launch(&session.stream).map_err(|e| e.to_string())?;
            }
            assert_eq!(
                download(session, &output)?,
                expected,
                "{q:?}, ids={ids:?}, replay={replay}"
            );
        }
        assert_eq!(download(session, &active)?, active_image);
    }
    assert_eq!(download(session, &bank)?, bank_image);
    Ok(())
}

#[test]
fn scale_expand_exact_bytes_and_guarded_graph_replay_on_gpu() {
    kernel_harness::gpu_test(128 << 20, |session| {
        let module = load(session)?;
        for (q, pattern) in cases() {
            run_case(session, &module, &fixture(q, pattern))?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn scale_expand_high_matrix_offsets_on_gpu() {
    // Expert 128 begins at 2 GiB, expert 256 at 4 GiB. Only their two
    // matrices are live. The encoded bank still fits its u32 offsets.
    kernel_harness::gpu_test(8_u64 << 30, |session| {
        let module = load(session)?;
        for width in [1, 2, 4] {
            let q = ScaleExpandQuery {
                width,
                rows: 64,
                groups: 1 << 18,
                projections: 1,
                experts: 257,
            };
            let matrix_bytes = (q.rows * q.groups) as usize;
            let fixed = codec::fixed_bytes(q.rows.into(), q.groups.into(), width).unwrap() as usize;
            let mut block = vec![0; fixed];
            block[..q.rows as usize].fill(127);
            // Exercise the largest legal 24-bit exception index as well.
            block.extend_from_slice(&((((matrix_bytes - 1) as u32) << 8) | 255).to_le_bytes());
            let layout = codec::bank_layout(&vec![block.len() as u64; q.experts as usize]).unwrap();
            let bank = session.allocate(layout.bytes as usize)?;
            poison::write(session, &bank, 0, &layout.table)?;
            // Reuse one uploaded block; avoid a multi-GiB host fixture.
            let encoded = upload(session, &block)?;
            for &offset in &layout.offsets {
                bank.copy_range_from_async(
                    offset as usize,
                    &encoded,
                    0,
                    block.len(),
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
            }
            let expected = decode_bank_matrix(
                &assemble(&[block]),
                0,
                q.rows.into(),
                q.groups.into(),
                width,
            )
            .unwrap();
            let active = upload(session, &active_bytes(q.experts, &[256, 128]))?;
            let raw_bytes = q.output_bytes().unwrap() as usize;
            let output = session.allocate(raw_bytes + OUTPUT_GUARD + TAIL_GUARD)?;
            let function = module
                .function(expand::symbol(width).unwrap())
                .map_err(|e| e.to_string())?;
            let config = launch::config(expand::plan(q).unwrap())?;
            let invoke = |stream: &Stream| {
                launch::launch_with(
                    &function,
                    stream,
                    config,
                    &[
                        KernelArgument::Buffer(&bank),
                        KernelArgument::Buffer(&active),
                        KernelArgument::Range {
                            buffer: &output,
                            offset: OUTPUT_GUARD,
                            byte_len: raw_bytes,
                        },
                        KernelArgument::U32(q.rows),
                        KernelArgument::U32(q.groups),
                        KernelArgument::U32(q.projections),
                        KernelArgument::U32(q.experts),
                    ],
                )
            };
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let graph = session.capture(invoke)?;
            for replay in [false, true] {
                poison::fill(
                    session,
                    &output,
                    std::slice::from_ref(&(0..output.byte_len())),
                    &[0xa5],
                )?;
                if replay {
                    graph.launch(&session.stream).map_err(|e| e.to_string())?;
                } else {
                    invoke(&session.stream)?;
                }
                // These include the low alias of a wrapped 4-GiB address
                // and the immediate unlisted neighbors of selected experts.
                for expert in [0, 127, 128, 129, 255, 256] {
                    let mut actual = vec![0; matrix_bytes];
                    output
                        .copy_to_bytes_at_async(
                            OUTPUT_GUARD + expert * matrix_bytes,
                            &mut actual,
                            &session.stream,
                        )
                        .map_err(|e| e.to_string())?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    if [128, 256].contains(&expert) {
                        assert_eq!(actual, expected, "width {width}, expert {expert}");
                    } else {
                        assert!(
                            actual.iter().all(|b| *b == 0xa5),
                            "unlisted expert {expert}"
                        );
                    }
                }
                for (offset, bytes) in [(0, OUTPUT_GUARD), (OUTPUT_GUARD + raw_bytes, TAIL_GUARD)] {
                    let mut guard = vec![0; bytes];
                    output
                        .copy_to_bytes_at_async(offset, &mut guard, &session.stream)
                        .map_err(|e| e.to_string())?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    assert!(guard.iter().all(|b| *b == 0xa5));
                }
            }
        }
        Ok(())
    })
    .unwrap();
}
